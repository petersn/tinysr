// This app does utterance detection and feature extraction, then saves out feature vectors.
// Use this app to collect data for training.

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
		printf("<command to produce audio> | store_utters <output directory>\n");
		printf("Does utterance detection, and saves each utterance to the output directory.\n");
		return 1;
	}

	// Allocate a context.
	fprintf(stderr, "Allocating context.\n");
	tinysr_ctx_t* ctx = tinysr_allocate_context();
	ctx->input_sample_rate = 16000;
	samp_t array[READ_SAMPS];
	keep_reading = 1;
	signal(SIGINT, sig_handler);
	while (keep_reading) {
		// Try to read in samples.
		size_t samples_read = fread(array, sizeof(samp_t), READ_SAMPS, stdin);
		if (samples_read == 0) break;
		// Feed in the samples to our recognizer.
		tinysr_feed_input(ctx, array, (int)samples_read);
		tinysr_detect_utterances(ctx);
		while (ctx->utterance_list.length) {
			// Try to find a free filename.
			int number = 0;
			char path[512];
			do {
				snprintf(path, sizeof(path), "%s/utter_%i.csv", argv[1], number++);
			} while (access(path, F_OK) != -1);
			fprintf(stderr, "Writing feature vectors to: '%s'\n", path);
			FILE* fp = fopen(path, "w");
			if (fp == NULL) {
				perror(path);
				return 2;
			}
			utterance_t* utterance = list_pop_front(&ctx->utterance_list);
			int i;
			for (i = 0; i < utterance->length; i++) {
				// Write the feature vector to stdout as CSV.
				feature_vector_t* fv = &utterance->feature_vectors[i];
				fprintf(fp, "%f", fv->log_energy);
				int j;
				for (j = 0; j < 13; j++)
					fprintf(fp, ",%f", fv->cepstrum[j]);
				fprintf(fp, "\n");
			}
			fclose(fp);
			free(utterance->feature_vectors);
			free(utterance);
		}
	}
	fprintf(stderr, "Freeing context. Processed %i samples.\n", ctx->processed_samples);
	tinysr_free_context(ctx);

	return 0;
}

