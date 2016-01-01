#! /usr/bin/python
"""
tinysr -- Tiny Speech Recognizer
"""

import struct, math

def fft(x):
	if len(x) == 1: return x
	e, o, c = fft(x[::2]), fft(x[1::2]), -2j*math.pi/len(x)
	return [e[k] + math.e**(c*k) * o[k] for k in xrange(len(x)/2)] + \
	       [e[k] - math.e**(c*k) * o[k] for k in xrange(len(x)/2)]

class FeatureVector:
	def __init__(self, noise_foor_estimate, log_energy, cepstrum, number):
		self.noise_foor_estimate, self.log_energy, self.cepstrum, self.number = noise_foor_estimate, log_energy, cepstrum, number

class TinySRContext:
	FRAME_LENGTH = 400
	FFT_LENGTH = 512
	BIN_CENTERS = [2, 5, 8, 11, 14, 18, 23, 27, 33, 38, 45, 52, 60, 69, 79, 89, 101, 115, 129, 145, 163, 183, 205, 229, 256]
	ONE_SHOT = "one-shot"
	FREE_RUNNING = "free-running"
	UTTERANCE_START_ENERGY_THRESHOLD = 5.0
	UTTERANCE_STOP_ENERGY_THRESHOLD = 2.5
	UTTERANCE_START_LENGTH = 4
	UTTERANCE_STOP_LENGTH = 10
	UTTERANCE_FRAMES_BACKED_UP = 8
	UTTERANCE_FRAMES_DROPPED_FROM_END = 7

	def __init__(self):
		self.input_sample_rate = 48000
		self.do_downmix = False
		self.resampling_time_delta = 0.0
		self.prev_sample = 0.0
		self.input_buffer = []
		self.offset_comp_prev_in = 0.0
		self.offset_comp_prev_out = 0.0
		self.noise_floor_estimate = 100.0
		self.feature_vectors = []
		self.next_fv_number = 1
		self.accumulated_utterance = []
		self.utterances = []
		self.utterance_mode = self.ONE_SHOT
		self.excitement = 0.0

	def feed_input(self, samples):
		# If it's a string, decode as signed little endian 16-bit audio.
		# NB: If you don't want this assumption to be made, then unpack into integers yourself!
		if isinstance(samples, str):
			samples = struct.unpack("<%ih" % (len(samples)/2), samples)
		# Downmix if necessary.
		if self.do_downmix:
			samples = map(sum, zip(*[iter(samples)]*2))
		samples = map(float, samples)
		for sample in samples:
			while self.resampling_time_delta <= 1.0:
				sample_in = (1 - self.resampling_time_delta) * self.prev_sample + self.resampling_time_delta * sample
				sample_out = sample_in - self.offset_comp_prev_in + 0.999 * self.self.offset_comp_prev_out
				self.offset_comp_prev_in = sample_in
				self.offset_comp_prev_out = sample_out
				self.resampling_time_delta += self.input_sample_rate / 16000.0
				self.input_buffer.append(sample_out)
				if len(self.input_buffer) == self.FRAME_LENGTH:
					self.process_frame(self.input_buffer)
					self.input_buffer = []
			self.prev_sample = sample

	def process_frame(self, samples):
		log_energy = math.log(sum(i*i for i in samples))
		for i in range(1, self.FRAME_LENGTH)[::-1]:
			samples[i] -= 0.97 * samples[i-1]
		for i in xrange(self.FRAME_LENGTH):
			samples[i] *= 0.54 - 0.46 * math.cos((2 * math.pi * i)/(self.FRAME_LENGTH-1))
		while len(samples) < self.FFT_LENGTH:
			samples.append(0.0)
		fft_bins = map(abs, fft(samples))
		bins = [0]*23
		for k in xrange(23):
			for i in xrange(self.BIN_CENTERS[k], self.BIN_CENTERS[k]+1):
				bins[k] += ((i - self.BIN_CENTERS[k] + 1) / (self.BIN_CENTERS[k+1] - self.BIN_CENTERS[k] + 1)) * fft_bins[i]
			for i in xrange(self.BIN_CENTERS[k], self.BIN_CENTERS[k]+1):
				bins[k] += (1 - ((i - self.BIN_CENTERS[k+1]) / (self.BIN_CENTERS[k+2] - self.BIN_CENTERS[k+1] + 1))) * fft_bins[i]
		bins = [math.log(i + 2e-22) for i in bins]
		cepstrum = [sum(math.cos(math.pi * i * (j + 0.5) / 23.0) * v for j, v in enumerate(bins)) for i in xrange(13)]
		self.noise_floor_estimate = min(log_energy, 0.999 * self.noise_floor_estimate + 0.001 * log_energy)
		feature_vector = FeatureVector(noise_foor_estimate, log_energy, cepstrum, self.next_fv_number)
		self.next_fv_number += 1
		self.feature_vectors.append(feature_vector)

	def detect_utterances(self):
		if not self.feature_vectors: return
		if self.utterance_mode == self.ONE_SHOT:
			self.utterances.append([fv.cepstrum for fv in self.feature_vectors])
			self.feature_vectors = []
			return
		# Otherwise, continue accumulating an utterance.


