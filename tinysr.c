// TinySR
// Written by Peter Schmidt-Nielsen (snp@mit.edu) starting in 2014.
// The entire project is placed in the public domain.
//
// Implements ES 201 108 feature extraction.

#include "tinysr.h"
#include <stdio.h>

// Defined here to avoid polluting the scope of the user.
#ifndef PI
#  define PI 3.141592653589793116
#endif
#ifndef PI2
#  define PI2 6.283185307179586232
#endif

void list_push(list_t* list, void* datum) {
	list->length++;
	list->tail = malloc(sizeof(list_node_t));
	list->tail->datum = datum;
	list->tail->prev = list->tail;
	list->tail->next = NULL;
	if (list->head == NULL)
		list->head = list->tail;
}

void* list_pop(list_t* list) {
	if (list->head == NULL)
		return NULL;
	list->length--;
	list_node_t* head = list->head;
	void* result = head->datum;
	list->head = head->next;
	free(head);
	if (list->head) list->head->prev = NULL;
	else list->tail = NULL;
	return result;
}

// Allocate a context for speech recognition.
tinysr_ctx_t* tinysr_allocate_context(void) {
	tinysr_ctx_t* ctx = malloc(sizeof(tinysr_ctx_t));
	// By default, assume 48000 frames per second of input.
	ctx->input_framerate = 48000;
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
	ctx->temp_buffer = malloc(sizeof(float) * FFT_LENGTH);
	// The features vector list is initially empty.
	ctx->fv_list = (list_t){0};
	return ctx;
}

// Frees a context and all associated memory.
void tinysr_free_context(tinysr_ctx_t* ctx) {
	free(ctx->input_buffer);
	free(ctx->temp_buffer);
	free(ctx);
}

// Feed in samples to the speech recognizer.
// Performs speech recognition immediately, as frames become complete.
void tinysr_feed_input(tinysr_ctx_t* ctx, samp_t* samples, int length) {
	while (length--) {
		// Read one sample into our input buffer.
		float sample_in = (float)*samples++;
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
	}
}

// Call to trigger recognition on all the accumulated frames.
void tinysr_recognize_frames(tinysr_ctx_t* ctx) {
	// Iterate over accumulated feature vectors.
	while (ctx->fv_list.length > 0) {
		feature_vector_t* fv = list_pop(&ctx->fv_list);
		printf("Log energy: %.2f\n", fv->log_energy);
		printf("Cepstrum:");
		int i;
		for (i = 0; i < 13; i++)
			printf(" %.2f", fv->cepstrum[i]);
		printf("\n");
		free(fv);
	}
}

// Private function: Do not call directly!
// Initiates recognition on the contents of ctx->input_buffer.
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
	// If these assumptions change, rerung that script to figure out what these bins should be!
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
	float cepstrum[13];
	for (i = 0; i < 13; i++) {
		cepstrum[i] = 0.0;
		// Compute the discrete cosine transform (DCT) the naive way.
		// XXX: Again notice that I'm zero indexing: filter_bank[j] contains what the spec calls f_(j+1).
		// This is why it's (j + 0.5) rather than (j - 0.5) like in the spec in the upcoming expression.
		int j;
		for (j = 0; j < 23; j++)
			cepstrum[i] += filter_bank[j] * cosf(PI * i * (j + 0.5) / 23.0);
	}
	// We're now done with the entire front-end processing!
	// Now we save the feature vector which consists of log_energy, and cepstrum into the list.
	feature_vector_t* fv = malloc(sizeof(feature_vector_t));
	fv->log_energy = log_energy;
	for (i = 0; i < 13; i++)
		fv->cepstrum[i] = cepstrum[i];
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
		// TODO, if FFTs become limiting: precompute these trig coefficients, and reuse them throughout the computation.
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

