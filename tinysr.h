// TinySR
// Written by Peter Schmidt-Nielsen (snp@mit.edu) starting in 2014.
// The entire project is placed in the public domain.

#ifndef TinySR_H
#define TinySR_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

#define FFT_LENGTH 512
#define FRAME_LENGTH 400
#define SHIFT_INTERVAL 160

typedef int16_t samp_t;

typedef enum {
	TINYSR_MODE_ONE_SHOT,
	TINYSR_MODE_FREE_RUNNING
} tinysr_mode_t;

// Generic singly linked list based stack.
typedef struct _list_node_t {
	void* datum;
	struct _list_node_t* prev;
	struct _list_node_t* next;
} list_node_t;

// Simply set each field to zero initially to make an empty list.
// Thus, list_t l = {0}; suffices for initializiation.
typedef struct {
	int length;
	list_node_t* head;
	list_node_t* tail;
} list_t;

void list_append_back(list_t* list, void* datum);
void* list_pop_front(list_t* list);

// TinySR context, and associated functions.
typedef struct {
	// Public configuration:
	// The sample rate you are feeding the recognizer.
	// It is safe to change this as frequently as you want.
	int input_sample_rate;
	tinysr_mode_t utterance_mode;
	int do_downmix;

	// Private:
	int processed_samples;
	float resampling_prev_raw_sample;
	float resampling_time_delta;
	float offset_comp_prev_in;
	float offset_comp_prev_out;
	float* input_buffer;
	int input_buffer_next;
	int input_buffer_samps;
	float* temp_buffer;
	// Feature vector list.
	list_t fv_list;
	long long next_fv_number;
	list_node_t* current_fv;
	list_node_t* utterance_start;
	float noise_floor_estimate;
	float excitement;
	float boredom;
	int utterance_state;
	list_t utterance_list;
	list_t recog_entry_list;
	char** word_names;
	list_t results_list;
} tinysr_ctx_t;

typedef struct {
	long long number;
	float log_energy;
	float cepstrum[13];
	float noise_floor;
} feature_vector_t;

typedef struct {
	int length;
	feature_vector_t* feature_vectors;
} utterance_t;

typedef struct {
	// The log-likelihood of data matching this model is:
	// log_likelihood_offset - 0.5 * (cepstrum - cepstrum_mean)^T * cepstrum_inverse_covariance * (cepstrum - cepstrum_mean)
	// Where cepstrum is an input 13-column vector and cepstrum_inverse_covariance is a 13x13 matrix.
	// Also note that covariance matrices are symmetric, so there is no row/column major order issue
	// to worry about with the cepstrum_inverse_covariance.
	float log_likelihood_offset;
	float cepstrum_mean[13];
	float cepstrum_inverse_covariance[169];
} gaussian_t;

typedef struct {
	int index;
	char* name;
	float ll_offset, ll_slope;
	int model_template_length;
	gaussian_t* model_template; 
} recog_entry_t;

typedef struct {
	int word_index;
	float score;
} result_t;

// === Public API ===

// Call to get/free a context.
tinysr_ctx_t* tinysr_allocate_context(void);
void tinysr_free_context(tinysr_ctx_t* ctx);

// Convenience function call, equivalent to tinysr_feed_input(), but then runs
// tinysr_detect_utterances() and tinysr_recognize_utterances(), and then returns
// the number of pending recognition results.
int tinysr_recognize(tinysr_ctx_t* ctx, samp_t* samples, int length);

// Call to pass input samples.
void tinysr_feed_input(tinysr_ctx_t* ctx, samp_t* samples, int length);

// Call to trigger utterance detection.
void tinysr_detect_utterances(tinysr_ctx_t* ctx);

// Call to trigger recognition on detected utterances.
void tinysr_recognize_utterances(tinysr_ctx_t* ctx);

// Call to get one recognition result.
// Returns 1 if a result was gotten, 0 otherwise.
// It's safe to set either or both pointers to NULL.
int tinysr_get_result(tinysr_ctx_t* ctx, int* word_index, float* score);

// Add some recognition entries.
// Call this to add a word to the vocabulary of the given context.
int tinysr_load_model(tinysr_ctx_t* ctx, const char* path);

// Read and write CSV files containing an utterance.
// The write function returns non-zero on error, but doesn't print anything.
int write_feature_vector_csv(const char* path, utterance_t* utterance);
utterance_t* read_feature_vector_csv(const char* path);

// === Private functions ===

// Initiates a feature extraction run on the contents of input_buffer.
// This function is called automatically by tinysr_feed_input whenever the
// buffer is full, so you should never have to call it yourself.
void tinysr_process_frame(tinysr_ctx_t* ctx);

void tinysr_recognize_utterance(tinysr_ctx_t* ctx, utterance_t* utterance);

float gaussian_log_likelihood(gaussian_t* gauss, feature_vector_t* fv);
float compute_dynamic_time_warping(recog_entry_t* match, utterance_t* utterance);

// This function is used internally for the FFT computation.
void tinysr_fft_dit(float* in_real, float* in_imag, int length, int stride, float* out_real, float* out_imag);
void tinysr_abs_fft(float* array, int length);

#ifdef __cplusplus
}
#endif

#endif

