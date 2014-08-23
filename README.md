TinySR: Tiny Speech Recognizer
==============================

**TinySR** is a light weight real-time small-vocabulary speaker-independent speech recognizer written in portable C.
The entire library fits in a single pair of files, `tinysr.c` and `tinysr.h`.
You can either statically link, dynamically link, or directly include the pair of files in your project, as TinySR is entirely public domain.
No special libraries are used or dependencies required; everything is here.

The goal is to provide the following features:
* Lean, readable source with absolutely no external dependencies.
* Very easy API for both real-time and one-shot recognition.
* Real-time low latency performance, and utterance detection.
* User friendly vocabulary training, and good speaker independence.

Currently the recognition pipeline looks like:
* Generic audio processing (resampling filter, bringing input to 16 kHz)
* ES 201 108 feature extraction, to log energy + 13 cepstral components.
* Utterance detection, followed by utterance level Cepstral Mean Normalization.
* Maximum likelihood multivariate Gaussian models.
* Dynamic Time Warping to match against the vocabulary.

If you use my code, I'd love it if you dropped me a line at <snp@mit.edu>.

The code is divided into the following directories:
* `apps`: Contains programs that link against `tinysr.o`. The makefile is set up to automatically compile anything matching `*.c` in `apps` against `tinysr.o`.
* `playground`: Contains non-critical throw-away programs that were written in the course of creating TinySR.
* `scripts`: Contains utility scripts, such as for speaker training, or computing important tables of constants.

