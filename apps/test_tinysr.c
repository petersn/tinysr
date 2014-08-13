// Test TinySR.

#include <stdio.h>
#include <math.h>
#include "tinysr.h"

#define COUNT 100
#define SIZE 512

int main(int argc, char** argv) {
	int i, j;
	float v[SIZE];
	printf("Running %i FFTs of size %i.\n", COUNT, SIZE);
	for (i = 0; i < COUNT; i++) {
		for (j = 0; j < SIZE; j++)
			v[j] = j;
		tinysr_abs_fft(v, SIZE);
	}

	printf("Allocating context.\n");
	tinysr_ctx_t* ctx = tinysr_allocate_context();
	// Set the framerate of the input audio we're passing it.
	// Choosing a funky value, just for fun.
	ctx->input_sample_rate = 7357;
	samp_t frame[500];
	for (i = 0; i < 500; i++)
		frame[i] = i + (int) (40 * cosf(i/0.6));
	tinysr_feed_input(ctx, frame, 500);
	tinysr_detect_utterances(ctx);
	printf("Freeing context.\n");
	tinysr_free_context(ctx);

	return 0;
}

