// This app runs utterance detection on audio streamed in via stdin.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <math.h>
#include "tinysr.h"

#define READ_SAMPS 128

int keep_reading;
void sig_handler(int signo) {
	printf("SIGINT caught, stopping.\n");
	keep_reading = 0;
}

int main(int argc, char** argv) {
	if (argc != 2) {
		printf("Usage:\n");
		printf("<command to produce audio> | full_reco <speech_model>\n");
		printf("Expects the input to be 16000 Hz mono 16-bit signed little endian raw audio.\n");
		printf("Expects a file called speech_model in the same directory.\n");
		return 1;
	}

	// Allocate a context.
	fprintf(stderr, "Allocating context.\n");
	tinysr_ctx_t* ctx = tinysr_allocate_context();
	ctx->input_sample_rate = 16000;
	ctx->utterance_mode = TINYSR_MODE_FREE_RUNNING;
	printf("Loaded up %i words.\n", tinysr_load_model(ctx, argv[1]));
	samp_t array[READ_SAMPS];
	keep_reading = 1;
	signal(SIGINT, sig_handler);
	int state = 0;
	while (keep_reading) {
		// Try to read in samples.
		size_t samples_read = fread(array, sizeof(samp_t), READ_SAMPS, stdin);
		if (samples_read == 0) break;
		// Feed in the samples to our recognizer.
		tinysr_recognize(ctx, array, (int)samples_read);
		if (state == 0 && ctx->utterance_state == 1)
			printf("Utterance detected.\n");
		if (state == 1 && ctx->utterance_state == 0)
			printf("Utterance over.\n");
		state = ctx->utterance_state;
		// Get back results.
		int word_index;
		float score;
		while (tinysr_get_result(ctx, &word_index, &score))
			printf("=== %s (%.3f)\n", ctx->word_names[word_index], score);
	}
	fprintf(stderr, "Freeing context. Processed %i samples.\n", ctx->processed_samples);
	tinysr_free_context(ctx);

	return 0;
}

