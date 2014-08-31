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
	if (argc != 2 || strcmp(argv[1], "--go")) {
		printf("Usage:\n");
		printf("<command to produce audio> | detect_utter --go\n");
		printf("Expects the input to be 16000 Hz mono 16-bit signed little endian raw audio.\n");
		printf("Does utterance detection, and prints out the results.\n");
		printf("Some example commands that can produce suitable audio:\n");
		printf("arecord -r 16000 -c 1 -f S16_LE\n");
		printf("ffmpeg -y -f alsa -ac 1 -i default -ar 16000 -f s16le -acodec pcm_s16le /dev/stdout\n");
		return 1;
	}

	// Allocate a context.
	fprintf(stderr, "Allocating context.\n");
	tinysr_ctx_t* ctx = tinysr_allocate_context();
	ctx->input_sample_rate = 16000;
	ctx->utterance_mode = TINYSR_MODE_FREE_RUNNING;
	samp_t array[READ_SAMPS];
	keep_reading = 1;
	signal(SIGINT, sig_handler);
	int state = 0;
	while (keep_reading) {
		// Try to read in samples.
		size_t samples_read = fread(array, sizeof(samp_t), READ_SAMPS, stdin);
		if (samples_read == 0) break;
		// Feed in the samples to our recognizer.
		tinysr_feed_input(ctx, array, (int)samples_read);
		tinysr_detect_utterances(ctx);
		if (state == 0 && ctx->utterance_state == 1) {
			printf("Utterance detected.\n");
			state = 1;
		}
		if (state == 1 && ctx->utterance_state == 0) {
			printf("Utterance over.\n");
			state = 0;
		}
	}
	fprintf(stderr, "Freeing context. Processed %i samples.\n", ctx->processed_samples);
	tinysr_free_context(ctx);

	return 0;
}

