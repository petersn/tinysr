// TinySR
// Written by Peter Schmidt-Nielsen (snp@mit.edu) starting in 2014.
// The entire project is placed in the public domain.
//
// Implements ES 201 108 feature extraction.

#include "tinysr.h"
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

// Defined here to avoid polluting the scope of the user.
#ifndef PI
#  define PI 3.141592653589793116
#endif
#ifndef PI2
#  define PI2 6.283185307179586232
#endif

// The log energy of the frame must exceed the noise floor estimate by this much to trigger excitement, which triggers utterance detection.
#define UTTERANCE_START_ENERGY_THRESHOLD 5.0
// Alternatively, the log energy must NOT exceed the noise floor by this much to trigger boredom, which ends an utterance.
#define UTTERANCE_STOP_ENERGY_THRESHOLD 1.0
// The state machine must get excitement this many frames in a row to trigger an utterance.
#define UTTERANCE_START_LENGTH 10
// And, to end an utterance, the state machine must get boredom this many frames in a row.
#define UTTERANCE_STOP_LENGTH 10
// When an utterance is detected, this many frames before the beginning of the detection
// are also scooped up. This is to take into account the fact that most utterances begin
// with quiet intro dynamics that are hard to pick up otherwise. Note: This doesn't take
// into account UTTERANCE_START_LENGTH, so if this value is zero, then UTTERANCE_START_LENGTH-1
// exciting frames will be missed.
#define UTTERANCE_FRAMES_BACKED_UP 15

void list_push(list_t* list, void* datum) {
	list->length++;
	// Create the new list node, and fill out its entries.
	list_node_t* tail = malloc(sizeof(list_node_t));
	tail->datum = datum;
	tail->prev = list->tail;
	tail->next = NULL;
	// Link it into the list.
	if (list->tail != NULL)
		list->tail->next = tail;
	list->tail = tail;
	if (list->head == NULL)
		list->head = list->tail;
}

void* list_pop(list_t* list) {
	if (list->head == NULL)
		return NULL;
	list->length--;
	// Find the head node.
	list_node_t* head = list->head;
	void* result = head->datum;
	list->head = head->next;
	free(head);
	if (list->head != NULL) list->head->prev = NULL;
	else list->tail = NULL;
	return result;
}

// Allocate a context for speech recognition.
tinysr_ctx_t* tinysr_allocate_context(void) {
	tinysr_ctx_t* ctx = malloc(sizeof(tinysr_ctx_t));
	ctx->processed_samples = 0;
	// Initialize the resampling filter.
	ctx->resampling_prev_raw_sample = 0.0;
	ctx->resampling_time_delta = 0.0;
	// By default, assume the input is at 48000 samples per second.
	ctx->input_sample_rate = 48000;
	// Offset compensation running values.
	ctx->offset_comp_prev_in = 0.0;
	ctx->offset_comp_prev_out = 0.0;
	// Allocate a circular buffer for staging the input.
	ctx->input_buffer = malloc(sizeof(float) * FRAME_LENGTH);
	// Index to write to next in input_buffer.
	ctx->input_buffer_next = 0;
	// How many samples are currently in input_buffer.
	ctx->input_buffer_samps = 0;
	// Allocate a temporary buffer for processing.
	// The entire feature extraction takes place in ths buffer, so we make it long enough to do an FFT in.
	ctx->temp_buffer = malloc(sizeof(float) * FFT_LENGTH);
	// The features vector list: whenever a frame of input is processed, the resultant features go in here.
	ctx->fv_list = (list_t){0};
	// The feature vectors are numbered in the list, and this variable stores the next value to be assigned.
	ctx->next_fv_number = 1;
	// This points to the fv_list node that has been most recently checked.
	// It might not be the end of fv_list if new frames have been added, but not yet processed.
	ctx->current_fv = NULL;
	// When an utterance is detected as beginning, this variable is set to point to the beginning FV.
	ctx->utterance_start = NULL;
	// The running estimate of the noise floor.
	// Initially set it to any over-estimate, essentially infinity.
	ctx->noise_floor_estimate = 100.0;
	// These values accumulate during energy and silence respectively, and reset to zero during the opposite.
	// They're floats, but in the current implementation they accumulate at a rate of 1.0 per feature vector.
	// They are used in the utterance detection state machine to determine starting and stoping respectively.
	ctx->excitement = 0.0;
	ctx->boredom = 0.0;
	// This variable holds the main state of the utterance detection state machine.
	// If it is zero, then we are waiting for an utterance to start.
	// If it's one, then an utterance is in progress.
	ctx->utterance_state = 0;

	return ctx;
}

// Frees a context and all associated memory.
void tinysr_free_context(tinysr_ctx_t* ctx) {
	free(ctx->input_buffer);
	free(ctx->temp_buffer);
	// Free any feature vectors that happen to be allocated at the time.
	while (ctx->fv_list.length)
		free(list_pop(&ctx->fv_list));
	free(ctx);
}

// Feed in samples to the speech recognizer.
// Performs feature extraction immediately, as frames become complete.
void tinysr_feed_input(tinysr_ctx_t* ctx, samp_t* samples, int length) {
	while (length--) {
		// Read one sample in.
		float raw_sample = (float)*samples++;
		ctx->processed_samples++;
		// Now we apply the resampling filter, resampling from ctx->input_sample_rate to 16000 samples per second.
		while (ctx->resampling_time_delta <= 1.0) {
			// Linearly interpolate the current sample.
			float sample_in = (1 - ctx->resampling_time_delta) * ctx->resampling_prev_raw_sample + ctx->resampling_time_delta * raw_sample;
			// Perform offset compensation (ES 201 108 4.2.3)
			float sample_out = sample_in - ctx->offset_comp_prev_in + 0.999 * ctx->offset_comp_prev_out;
			ctx->offset_comp_prev_in = sample_in;
			ctx->offset_comp_prev_out = sample_out;
			// Store the sample into the circular buffer.
			ctx->input_buffer[ctx->input_buffer_next++] = sample_out;
			ctx->input_buffer_next %= FRAME_LENGTH;
			// Check if this completes a frame. (ES 201 108 4.2.4)
			if (++ctx->input_buffer_samps == FRAME_LENGTH) {
				tinysr_process_frame(ctx);
				ctx->input_buffer_samps -= SHIFT_INTERVAL;
			}
			// Advance our time estimate by the appropriate amount.
			ctx->resampling_time_delta += ctx->input_sample_rate / 16000.0;
		}
		// Store the current sample, for linear interpolation next time around.
		ctx->resampling_prev_raw_sample = raw_sample;
		ctx->resampling_time_delta -= 1;
	}
}

// Call to trigger utterance detection on all the accumulated frames.
void tinysr_recognize_frames(tinysr_ctx_t* ctx) {
	// If no feature vectors have yet been produced, we can't start processing.
	if (ctx->fv_list.length == 0)
		return;
	while (1) {
		// Try to get a new feature vector to process, either by starting up, or getting the next.
		if (ctx->current_fv == NULL)
			ctx->current_fv = ctx->fv_list.head;
		else if (ctx->current_fv->next != NULL)
			ctx->current_fv = ctx->current_fv->next;
		// If we can't get a new feature vector to process, we're done.
		// This guarantees that we only get past this line once for each feature vector.
		else break;
		// Now we processes this new feature vector.
		feature_vector_t* fv = (feature_vector_t*) ctx->current_fv->datum;
		static int counter = 0;
		if (counter++ % 30 == 0)
			printf(":: %.2f - %.2f\n", fv->noise_floor, fv->log_energy);
		// If the new FV's energy exceeds the threshold, become more excited. Otherwise, reset.
		if (fv->log_energy > fv->noise_floor + UTTERANCE_START_ENERGY_THRESHOLD)
			ctx->excitement += 1.0;
		else
			ctx->excitement = 0.0;
		if (fv->log_energy < fv->noise_floor + UTTERANCE_STOP_ENERGY_THRESHOLD)
			ctx->boredom += 1.0;
		else
			ctx->boredom = 0.0;
		// Here begins the utterance detection state machine. The state is stored in ctx->utterance_state.
		// Zero means waiting for utterance, one means waiting for utterance to end.
		if (ctx->utterance_state == 0) {
			// If we've become excited some number of feature vectors in a row, then we detect an utterance.
			if (ctx->excitement >= UTTERANCE_START_LENGTH) {
				printf("Utterance detected.\n");
				ctx->utterance_state = 1;
				ctx->utterance_start = ctx->current_fv;
				// Now back up some number of FVs. (See #defs at top for explanation.)
				int i;
				for (i = 0; i < UTTERANCE_FRAMES_BACKED_UP; i++)
					if (ctx->utterance_start->prev != NULL)
						ctx->utterance_start = ctx->utterance_start->prev;
			}
		} else if (ctx->boredom >= UTTERANCE_STOP_LENGTH) {
			printf("Utterance over.\n");
			// Initiate recognition on the detected utterance.
			tinysr_process_utterance(ctx);
			// Finally, reset our state machine.
			ctx->utterance_start = NULL;
			ctx->utterance_state = 0;
		}
	}
	// Now that we're done processing FVs for the time being, forget about old ones that no longer could
	// possibly be used in an utterance. Begin by computing the oldest possible FV number we could care about.
	long long oldest_still_relevant = 0;
	// We care about UTTERANCE_FRAMES_BACKED_UP frames before the current FV.
	if (ctx->current_fv != NULL)
		oldest_still_relevant = ((feature_vector_t*)ctx->current_fv->datum)->number - UTTERANCE_FRAMES_BACKED_UP;
	// We also care about any FVs currently in an utterance being detected.
	if (ctx->utterance_start != NULL)
		oldest_still_relevant = ((feature_vector_t*)ctx->utterance_start->datum)->number;
	// While the oldest FV in the list is too old to be relevant, drop it.
	while (ctx->fv_list.length && ((feature_vector_t*)ctx->fv_list.head->datum)->number < oldest_still_relevant)
		free(list_pop(&ctx->fv_list));
}

// This function reads the current state, and processes an utterance.
// Do not call directly! This function requires that the ctx have some variables
// loaded with appropriate values to point to the data of the utterance.
// Specifically, ctx->utterance_start and ctx->current_fv must point to the
// first and last feature vector in the utterance, respectively and inclusively.
void tinysr_process_utterance(tinysr_ctx_t* ctx) {
	// Count the length of the utterance, by traversing the singly linked list.
	int utterance_length = 1;
	list_node_t* node;
	for (node = ctx->current_fv; node != ctx->current_fv; node = node->next)
		utterance_length++;
	// Copy over the utterance into a flat array, for processing.
	feature_vector_t* utterance = malloc(sizeof(feature_vector_t) * utterance_length);
	int i = 0;
	for (node = ctx->current_fv; node != ctx->current_fv; node = node->next)
		utterance[i++] = *(feature_vector_t*)node->datum;
	// Begin processing the utterance.
	// (1) Cepstral Mean Normalization: start by averaging the cepstrum over the utterance.
	float cepstral_mean[13] = {0};
	int j;
	for (i = 0; i < utterance_length; i++)
		for (j = 0; j < 13; j++)
			cepstral_mean[j] += utterance[i].cepstrum[j] / (float) utterance_length;
	// Then, subtract out the cepstral mean from the whole utterance.
	for (i = 0; i < utterance_length; i++)
		for (j = 0; j < 13; j++)
			utterance[i].cepstrum[j] -= cepstral_mean[j];
	// (2) Perform Dynamic Time Warping against the model list.
	// TODO
	// Finally, free the utterance copy.
	free(utterance);
}

// Private function: Do not call directly!
// Initiates front-end feature extraction on the contents of ctx->input_buffer.
void tinysr_process_frame(tinysr_ctx_t* ctx) {
	int i;
	// Copy over the frame from the circular buffer into temp_buffer.
	// Currently the frame could be laid out in input_buffer like:
	// [ 6 7 8 9 0 1 2 3 4 5 ]
	//           ^ input_buffer_next
	// We straighten out this circular representation into temp_buffer.
	// Completing ES 201 108 4.2.4.
	int index = ctx->input_buffer_next;
	for (i = 0; i < FRAME_LENGTH; i++) {
		ctx->temp_buffer[i] = ctx->input_buffer[index++];
		index %= FRAME_LENGTH;
	}
	// Measure log energy. (ES 201 108 4.2.5)
	// Add a noise floor, keeping the log energy above -50.
	// (Slight deviation from spec, but makes almost no difference.)
	float energy = 2e-22;
	for (i = 0; i < FRAME_LENGTH; i++)
		energy += ctx->temp_buffer[i] * ctx->temp_buffer[i];
	float log_energy = logf(energy);
	// Pre-emphasize. (ES 201 108 4.2.6)
	for (i = FRAME_LENGTH-1; i > 0; i--)
		ctx->temp_buffer[i] -= 0.97 * ctx->temp_buffer[i-1];
	// The spec doesn't specify what happens to the first sample, so we just zero it.
	ctx->temp_buffer[0] = 0.0;
	// Hamming window. (ES 201 108 4.2.7)
	for (i = 0; i < FRAME_LENGTH; i++)
		ctx->temp_buffer[i] *= 0.54 - 0.46 * cosf((PI2 * i)/(FRAME_LENGTH-1));
	// Absolute value (complex magnitude) of FFT of the data, zero padded out to FFT_LENGTH. (ES 201 108 4.2.8)
	// First, zero pad.
	for (i = FRAME_LENGTH; i < FFT_LENGTH; i++)
		ctx->temp_buffer[i] = 0.0;
	// Then take the abs fft.
	tinysr_abs_fft(ctx->temp_buffer, FFT_LENGTH);
	// We now only proceed on ctx->temp_buffer[0 ... FFT_LENGTH/2] inclusive (inclusive means one more
	// sample than half!) because Hermitian symmetry makes the upper half data redundant.
	// Compute the triangular filter bank, a.k.a. Mel filtering. (ES 201 108 4.2.9)
	float filter_bank[23];
	// This next line has data computed by scripts/compute_mel_bins.py, assuming 512 FFT bins, and 16 kHz sampling rate.
	// If these assumptions change, rerun that script to figure out what these bins should be!
	int cbin[25] = {2, 5, 8, 11, 14, 18, 23, 27, 33, 38, 45, 52, 60, 69, 79, 89, 101, 115, 129, 145, 163, 183, 205, 229, 256};
	// XXX: Note! ES 201 108 has fbank (corresponding to our filter_bank) being one indexed, but I have it zero indexed.
	// Thus, note that cbin[k+1] is the center bin index for filter_bank[k]. This is why cbin is of length 25. 
	// The first and last bin indexes are for sizing the first and last triangular filter. Therefore, note that
	// where in 4.2.9 it says fbank_k is based on cbin_(k-1), cbin_k and cbin_(k+1), instead for me it's based on
	// cbin_k, cbin_(k+1), and cbin_(k+2). Just keep track off the off by oneness.
	int k;
	for (k = 0; k < 23; k++) {
		filter_bank[k] = 0.0;
		for (i = cbin[k]; i <= cbin[k+1]; i++)
			filter_bank[k] += ((i - cbin[k] + 1) / (float)(cbin[k+1] - cbin[k] + 1)) * ctx->temp_buffer[i];
		for (i = cbin[k+1]+1; i <= cbin[k+2]; i++)
			filter_bank[k] += (1 - ((i - cbin[k+1]) / (float)(cbin[k+2] - cbin[k+1] + 1))) * ctx->temp_buffer[i];
	}
	// Non-linear transform: logarithm. (ES 201 108 4.2.10)
	// Again note the noise floor of 2e-22 to prevent an answer less than -50.
	for (k = 0; k < 23; k++)
		filter_bank[k] = logf(filter_bank[k] + 2e-22);
	// Compute the mel cepstrum. (ES 201 108 4.2.11)
	float cepstrum[13] = {0};
	for (i = 0; i < 13; i++) {
		// Compute the discrete cosine transform (DCT) the naive way.
		// XXX: Again notice that I'm zero indexing: filter_bank[j] contains what the spec calls f_(j+1).
		// This is why it's (j + 0.5) rather than (j - 0.5) like in the spec in the upcoming expression.
		int j;
		for (j = 0; j < 23; j++)
			cepstrum[i] += filter_bank[j] * cosf(PI * i * (j + 0.5) / 23.0);
	}
	// Do noise floor estimation. Clearly, it's impossible for there to be less energy than the true noise floor.
	// Thus, if the energy is lower than our current floor estimate, then lower our estimate. However, if the
	// energy is greater than our estimate, raise it slowly. This is a ``slow to rise, fast to fall'' estimator.
	// We use 0.999 * old + 0.001 * new, which gives a ten second time constant with one frame per 10 ms. 
	if (log_energy < ctx->noise_floor_estimate) ctx->noise_floor_estimate = log_energy;
	else ctx->noise_floor_estimate = 0.999 * ctx->noise_floor_estimate + 0.001 * log_energy;
	// We're now done with the entire front-end processing!
	// Now we save the feature vector which consists of log_energy, and cepstrum into the list.
	feature_vector_t* fv = malloc(sizeof(feature_vector_t));
	fv->log_energy = log_energy;
	for (i = 0; i < 13; i++)
		fv->cepstrum[i] = cepstrum[i];
	// Consecutively number the feature vectors.
	fv->number = ctx->next_fv_number++;
	// Store the noise floor, so the utterance detector can take it into account.
	fv->noise_floor = ctx->noise_floor_estimate;
	list_push(&ctx->fv_list, fv);
}

// Computes the FFT on strided data recursively via decimation in time.
// This FFT should be equivalent to the pseudo-Python:
//     for k in xrange(length):
//         out[k] = sum(in[n] * math.e**(-2j * math.pi * n * k / length) for n in xrange(length))
void tinysr_fft_dit(float* in_real, float* in_imag, int length, int stride, float* out_real, float* out_imag) {
	if (length == 1) {
		*out_real = *in_real;
		*out_imag = *in_imag;
		return;
	}
	// Decimate in time.
	float even_real[length/2], even_imag[length/2], odd_real[length/2], odd_imag[length/2];
	tinysr_fft_dit(in_real, in_imag, length/2, stride*2, even_real, even_imag);
	tinysr_fft_dit(in_real + stride, in_imag + stride, length/2, stride*2, odd_real, odd_imag);
	// Do butterflies.
	int k;
	for (k = 0; k < length/2; k++) {
		float angle = -PI2 * k / (float)length;
		// TODO, if FFTs become limiting: precompute these trig coefficients, and reuse them.
		float coef_real = cosf(angle), coef_imag = sinf(angle);
		out_real[k]          = even_real[k] + coef_real * odd_real[k] - coef_imag * odd_imag[k];
		out_imag[k]          = even_imag[k] + coef_real * odd_imag[k] + coef_imag * odd_real[k];
		out_real[k+length/2] = even_real[k] - coef_real * odd_real[k] + coef_imag * odd_imag[k];
		out_imag[k+length/2] = even_imag[k] - coef_real * odd_imag[k] - coef_imag * odd_real[k];
	}
}

// Computes the absolute value of the FFT of an array of data.
void tinysr_abs_fft(float* array, int length) {
	assert(length > 0 && !(length & (length - 1))); // tinysr_abs_fft: array length must be power of two
	int i;
	// Allocate an array of zero imaginary parts.
	float imag[length];
	for (i = 0; i < length; i++)
		imag[i] = 0.0;
	// Take the FFT.
	tinysr_fft_dit(array, imag, length, 1, array, imag);
	// Take the absolute value of each entry.
	for (i = 0; i < length; i++)
		array[i] = sqrtf(array[i]*array[i] + imag[i]*imag[i]);
}

