TinySR: Tiny Speech Recognizer
==============================

**TinySR** is a light weight small-vocabulary speaker-independent speech recognizer written in portable C.
The entire library fits in a single pair of files, `tinysr.c` and `tinysr.h`.
You can either statically link, dynamically link, or directly include the pair of files in your project, as TinySR is entirely public domain.
No special libraries are used or dependencies required; everything is here.

Currently the recognition pipeline looks like:
* Generic audio processing (reconfigurable AGC + resampling filter, bringing input to 16 kHz)
* ES 201 108 feature extraction, to log energy + 13 cepstral components.
* Gaussian mixture models. (Unimplemented.)
* HMM with dynamic programming trellis decoder. (Unimplemented.)

The code is divided into the following directories:
* `apps`: Contains programs that link against `tinysr.o`. The makefile is set up to automatically compile anything matching `*.c` in `apps` against `tinysr.o`.
* `playground`: Contains non-critical throw-away programs that were written in the course of creating TinySR.
* `scripts`: Contains utility scripts, such as for speaker training, or computing important tables of constants.

