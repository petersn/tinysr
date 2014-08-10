// Test TinySR.

#include <stdio.h>
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
	// Setting it to the native 16 kHz that is used for recognition
	// causes no filtering to occur.
	ctx->input_framerate = 16000;
	samp_t frame[400];
	tinysr_feed_input(ctx, frame, 400);
	printf("Freeing context.\n");
	tinysr_free_context(ctx);

	return 0;
}

