TinySR: Tiny Speech Recognizer
==============================

**TinySR** is a light weight small-vocabulary speaker-independent speech recognizer written in portable C.
The entire library fits in a single pair of files, `tinysr.c` and `tinysr.h`.
You can either statically link, dynamically link, or directly include the pair of files in your project, as TinySR is entirely public domain.
Absolutely no external libraries are used.

Currently the recognition pipeline looks like:
* Generic audio processing (reconfigurable AGC + resampling filter, bringing input to 16 kHz)
* ES 201 108 feature extraction, to log energy + 13 cepstral components.
* Gaussian mixture models. (Unimplemented.)
* HMM with dynamic programming trellis decoder. (Unimplemented.)

