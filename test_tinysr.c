// Test TinySR.

#include <stdio.h>
#include "tinysr.h"

#define COUNT 10000
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
	return 0;
}


