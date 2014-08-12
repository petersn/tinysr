// This app is for training purposes.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tinysr.h"

#define READ_SAMPS 512

int main(int argc, char** argv) {
	if (argc != 3) {
		printf("Usage: compute_fv <sample rate> <input file>\n");
		printf("Expects the input to be raw 16-bit signed little endian audio at the sample rate.\n");
		printf("Computes feature vectors, and prints them out as CSV.\n");
		printf("Format is: \"log energy,cepstrum0,cepstrum1,...cepstrum12\\n\"\n");
		return 1;
	}

	// Allocate a context.
	fprintf(stderr, "Allocating context.\n");
	tinysr_ctx_t* ctx = tinysr_allocate_context();
	ctx->input_sample_rate = atoi(argv[1]);
	fprintf(stderr, "Reading as sample rate: %i\n", ctx->input_sample_rate);
	// Open the input file for reading.
	FILE* fp = fopen(argv[2], "rb");
	samp_t array[READ_SAMPS];
	while (1) {
		// Try to read in samples.
		size_t samples_read = fread(array, sizeof(samp_t), READ_SAMPS, fp);
		if (samples_read == 0) break;
		// Feed in the samples to our recognizer.
		tinysr_feed_input(ctx, array, (int)samples_read);
		// Reach into the context internals to pull out the feature vector.
		while (ctx->fv_list.length) {
			// Write the feature vector to stdout as CSV.
			feature_vector_t* fv = list_pop_front(&ctx->fv_list);
			printf("%f", fv->log_energy);
			int i;
			for (i = 0; i < 13; i++)
				printf(",%f", fv->cepstrum[i]);
			printf("\n");
			free(fv);
		}
	}
	fclose(fp);
	fprintf(stderr, "Freeing context. Processed %i samples.\n", ctx->processed_samples);
	tinysr_free_context(ctx);

	return 0;
}

