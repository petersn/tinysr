#! /usr/bin/python

from cmath import exp, pi

# FFT from rosetta code for comparison. 
def fft(x):
    N = len(x)
    if N <= 1: return x
    even = fft(x[0::2])
    odd =  fft(x[1::2])
    return [even[k] + exp(-2j*pi*k/N)*odd[k] for k in xrange(N/2)] + \
           [even[k] - exp(-2j*pi*k/N)*odd[k] for k in xrange(N/2)]

# Naive DFT, for making sure my numerics come out exactly the same.
def ft(x):
	return [sum(x[n] * exp(-2j*pi*n*k/len(x)) for n in xrange(len(x))) for k in xrange(len(x))]

# My FFT translation of the rosetta code.
# The arrays should be though of as being "allocated on the stack".
# Written to make it easy to translate into C.
def my_fft(A, start, stride, length, B):
	if length == 1:
		B[0] = A[start]
		return
	# Even FFT.
	even = [None]*(length/2)
	my_fft(A, start, stride*2, length/2, even)
	# Odd FFT.
	odd = [None]*(length/2)
	my_fft(A, start+stride, stride*2, length/2, odd)
	for k in xrange(length/2):
		coef = exp(-2j*pi*k/length)
		B[k] = even[k] + coef * odd[k]
		B[k+length/2] = even[k] - coef * odd[k]

# Some tests.
D = [1,2,3,4]
# These two should be equal.
print fft(D)
print ft(D)
# This should be 16 times D.
print ft(ft(ft(ft(D))))
# Finally, my_fft should also be equivalent.
O = [None] * len(D)
my_fft(D, 0, 1, len(D), O)
print O


