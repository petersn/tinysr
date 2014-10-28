TinySR: Tiny Speech Recognizer
==============================

![TinySR pipeline](http://web.mit.edu/snp/www/tinysr_diagram.png)

**TinySR** is a light weight real-time small-vocabulary speech recognizer written in portable C.
The entire library fits in a single pair of files, `tinysr.c` and `tinysr.h`, with approximately 500 source lines of code.
You can either statically link, dynamically link, or directly include the pair of files in your project, as TinySR is entirely public domain.
No special libraries are used or dependencies required; everything is here.

The goal is to provide the following features:
* Lean, readable source with absolutely no dependencies beyond the C standard library.
* Very easy API for recognition -- only five function calls required to do recognition! (Allocate, load model, pass in input, get out results, free.)
* Real-time low latency performance.
* User friendly vocabulary training.

Currently the recognition pipeline looks like:
* Generic audio processing (resampling filter, bringing input to 16 kHz)
* ES 201 108 feature extraction, to log energy + 13 cepstral components.
* Utterance detection, followed by utterance level Cepstral Mean Normalization.
* Maximum likelihood multivariate Gaussian models.
* Dynamic Time Warping to match against the vocabulary.

If you use my code, I'd love it if you dropped me a line at <snp@mit.edu>.
There's a LaTeX file in `docs` describing how TinySR works, and the compiled PDF is here: http://www.mit.edu/~snp/tinysr.pdf

The code is divided into the following directories:
* `apps`: Contains programs that link against `tinysr.o`. The makefile is set up to automatically compile anything matching `*.c` in `apps` against `tinysr.o`.
* `playground`: Contains non-critical throw-away programs that were written in the course of creating TinySR.
* `scripts`: Contains utility scripts, such as for speaker training, or computing important tables of constants.
* `pytinysr`: Contains a compatible recognizer implemented in pure Python, using no non-standard libraries. (Incomplete.)

The API
-------

To do recognition, all you need to do is:

```C
// Allocate a context.
tinysr_ctx_t* ctx = tinysr_allocate_context();
ctx->input_sample_rate = 44100;
// Load the acoustic model, which represents the vocabulary.
tinysr_load_model(ctx, "path/to/model");
// Process an audio buffer full of signed 16-bit little endian samples.
tinysr_recognize(ctx, audio_buffer, num_samples);
int word_index;
float score;
// Get the best matching word, and the score (higher = better) of the match.
tinysr_get_result(ctx, &word_index, &score);
// ctx->word_names maps vocabulary word indices to strings of their names.
printf("Word: %s (score: %f)\n", ctx->word_names[word_index], score);
// And, of course, free all allocated memory.
tinysr_free_context(ctx);
```

The only expectation is that `audio_buffer` is an array of signed 16-bit little endian samples, representing the input mono audio stream.
(To process stereo instead, set `ctx->do_downmix`.)
The sample rate of the buffer can be declared by setting `ctx->input_sample_rate` appropriately.
By default TinySR operates in "one shot" mode, in which the entire input is assumed to be a single utterance.
In a streaming application where utterance boundaries are not known a priori, TinySR can be switched to "free running" mode, by adding the following before the call to `tinysr_recognize`:

```C
ctx->utterance_mode = TINYSR_MODE_FREE_RUNNING;
```

To Train
--------

Before you can start training, you are going to need to produce a bunch of raw 16-bit 16 kHz mono audio.
On systems supporting ALSA, I recommend the following command to produce such raw audio, and stream it to stdout:

	arecord -r 16000 -c 1 -f S16_LE

Alternatively, if you have an audio file you'd like to use, the following command will convert it to appropriate raw audio and stream it to stdout:

	ffmpeg -i <file> -y -ar 16000 -ac 1 -f s16le -acodec pcm_s16le /dev/stdout

We will assume you have decided upon such a command, and will refer to it as `<audio>`.
To build a speech model for TinySR, begin by deciding on your vocabulary.
Once you've decided, make a directory for each word.
In this example, "up" and "down" are the vocabulary.
First, run:

	<audio> | ./apps/store_utters.app data/up

Say the word "up" at least twenty times, but preferably several times as many times, with long enough spaces in between that the app doesn't get confused.
Naturally, repeat this process for "down" into a separate directory.
Once you're done with this, run:

	python scripts/model_gen.py data/up data/down speech_model

You can now test the recognizer on your model by running:

	<audio> | ./apps/full_reco.app speech_model

The words will be printed to you based on the names of the directories containing their utterances as passed to `model_gen.py`.
Alternatively, if you're using the library's API, the names will be available in a table, but also as unambiguous indices.

Finally, some advice on building models.
If your goal is some degree of speaker independence, then I recommend that you produce separate male and female models for each word.
TinySR doesn't (yet) implement VTLN, so it's really crucial to get some good vocal tract length coverage across your training corpus.
Mixing the male and female speakers together during training works too, but greatly increases the variance of the Gaussians in the model.

Python Implementation
---------------------

**(Incomplete)**
Additionally, in `pytinysr` you will find `pytinysr.py`, which is a compatible implementation of TinySR in Python that can load the same acoustic model files and works identically.

To Do
-----

Some features I'm aiming for, or considering.
Drop me a line if you'd like to see one of these done sooner.

* Vocal Tract Length Normalization (VTLN).
* True Gaussian Mixture Models, with EM training, instead of the current single Gaussians. (Will make training much slower.)
* Differential features.
* Word error rate benchmarking app, for model validation.
* Maybe a 32-bit integer fixed precision only implementation for ARMs without FPUs?

