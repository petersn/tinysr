// TinySR
// Implements ES 201 108 feature extraction.

#include "tinysr.h"

#define PI2 6.283185307179586232

tinysr_ctx_t* tinysr_allocate_ctx(void) {
	tinysr_ctx_t* ctx = malloc(sizeof(tinysr_ctx_t));
	// By default, assume 48000 frames per second of input.
	ctx->input_framerate = 48000;
	// Offset compensation running values.
	ctx->offset_comp_prev_in = 0.0;
	ctx->offset_comp_prev_out = 0.0;
	// Allocate a circular buffer for staging the input.
	ctx->input_buffer = malloc(sizeof(float) * FFT_LENGTH);
	// Index to write to next in input_buffer.
	ctx->input_buffer_next = 0;
	// How many samples are currently in input_buffer.
	ctx->input_buffer_samps = 0;
	// Allocate a temporary buffer for processing.
	ctx->temp_buffer = malloc(sizeof(float) * FRAME_LENGTH);
	// Allocate a windowing buffer for storing the Hamming window to multiply against.
	ctx->windowing_buffer = malloc(sizeof(float) * FRAME_LENGTH);
	// The frame list is currently empty.

	// Precompute the Hamming window.
	int i;
	for (i = 0; i < FRAME_LENGTH; i++)
		ctx->windowing_buffer[i] = 0.54 - 0.46 * cosf((PI2 * i)/(FRAME_LENGTH-1));

	return ctx;
}

void tinysr_free_ctx(tinysr_ctx_t* ctx) {
	free(ctx->input_buffer);
	free(ctx->temp_buffer);
	free(ctx->windowing_buffer);
	free(ctx);
}

void tinysr_feed_input(tinysr_ctx_t* ctx, int length, samp_t* samples) {
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

void tinysr_process_frame(tinysr_ctx_t* ctx) {
	int i;
	// Copy over the frame from the circular buffer into temp_buffer.
	// Currently the frame could be laid out in input_buffer like:
	// [ 6 7 8 9 0 1 2 3 4 5 ]
    //           ^ input_buffer_next
	// We straighten out this circular representation.
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
	float logE = logf(energy);
	// Pre-emphasize. (ES 201 108 4.2.6)
	for (i = FRAME_LENGTH-1; i > 0; i--)
		ctx->temp_buffer[i] -= 0.97 * ctx->temp_buffer[i-1];
	// The spec doesn't specify what happens to the first sample, so we just zero it.
	ctx->temp_buffer[0] = 0.0;
	// Hamming window. (ES 201 108 4.2.7)
	for (i = 0; i < FRAME_LENGTH; i++)
		ctx->temp_buffer[i] *= ctx->windowing_buffer[i];
	// Absolute value (complex magnitude) of FFT of the data, zero padded out to FFT_LENGTH. (ES 201 108 4.2.8)
	// First, zero pad.
	for (i = FRAME_LENGTH; i < FFT_LENGTH; i++)
		ctx->temp_buffer[i] = 0.0;
	// Then take the abs fft.
	tinysr_abs_fft(ctx->temp_buffer, FFT_LENGTH);
}

// Computes the FFT on strided data recursively via decimation in time.
void tinysr_fft_dit(float* in_real, float* in_imag, int length, int stride, float* out_real, float* out_imag) {
	if (length == 1) {
		*out_real = *in_real;
		*out_imag = *in_imag;
		return;
	}
	// Decimate in time.
	float even_real[length/2];
	float even_imag[length/2];
	float odd_real[length/2];
	float odd_imag[length/2];
	tinysr_fft_dit(in_real, in_imag, length/2, stride*2, even_real, even_imag);
	tinysr_fft_dit(in_real + stride, in_imag + stride, length/2, stride*2, odd_real, odd_imag);
	// Do butterflies.
	int k;
	for (k = 0; k < length/2; k++) {
		float angle = -PI2 * k / (float)length;
		// Todo: Precompute these trig coefficients, and reuse them throughout the computation.
		float coef_real = cosf(angle);
		float coef_imag = sinf(angle);
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

