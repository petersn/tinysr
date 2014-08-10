// TinySR

#ifndef TinySR_HEADER_H
#define TinySR_HEADER_H

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#define FFT_LENGTH 512
#define FRAME_LENGTH 400
#define SHIFT_INTERVAL 160

typedef int64_t samp_t;

typedef struct _frame_list_t {
	float* frame;
	struct _frame_list_t* next;
} frame_list_t;

typedef struct {
	// Public configuration:
	int input_framerate;

	// Private:
	float offset_comp_prev_in;
	float offset_comp_prev_out;
	float* input_buffer;
	int input_buffer_next;
	int input_buffer_samps;
	float* temp_buffer;
	float* windowing_buffer;
	frame_list_t* frame_list_head;
	frame_list_t* frame_list_tail;
} tinysr_ctx_t;

// Call to get/free a context.
tinysr_ctx_t* tinysr_allocate_ctx(void);
void tinysr_free_ctx(tinysr_ctx_t* ctx);

// Call to pass input samples.
void tinysr_feed_input(tinysr_ctx_t* ctx, int length, samp_t* samples);

// Private functions.
// Initiates a feature extraction run on the contents of input_buffer.
// This function is called automatically by tinysr_feed_input whenever the
// buffer is full, so you should never have to call it yourself.
void tinysr_process_frame(tinysr_ctx_t* ctx);

// This function is used internally for the FFT computation.
void tinysr_fft_dit(float* in_real, float* in_imag, int length, int stride, float* out_real, float* out_imag);
void tinysr_abs_fft(float* array, int length);

#endif
