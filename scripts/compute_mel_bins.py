#! /usr/bin/python
"""
Compute the bin indices, as required by ES 201 108 4.2.9.
"""

import math

# These functions are as defined in the spec.

def Mel(x):
	return 2595 * math.log(1 + x/700.0, 10)

def Mel_inv(x):
	return (10 ** (x / 2595.0) - 1) * 700.0

f_start = 64.0
f_s = 16000.0
FFTL = 512

def center(i):
	return Mel_inv(Mel(f_start) + i * (Mel(f_s / 2) - Mel(f_start))/(23.0 + 1.0))

def to_bin(f):
	return int(round(f * FFTL / f_s))

# Compute the center frequencies.
f_c = map(center, range(1, 24))
print "Center frequencies:", f_c
# Compute the bins.
cbins = map(to_bin, f_c)
# Prepend and append the bins corresponding to the starting frequency and half the sampling frequency respectively.
cbins = [to_bin(f_start)] + cbins + [to_bin(f_s/2)]
# These are the FFT bin indices, cbin, as written in ES 201 108 4.2.9.
print "cbin =", cbins
