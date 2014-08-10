#! /usr/bin/python
# Quick testing code to determine if I'm happy with linear interpolation for audio resampling.
# My tests say, probably yes. The results sound quite reasonable.

import math, wave, struct, sys

# Acheive resampling via linear interpolation.
def resample(a, old_rate, new_rate):
	print "Resampling from %i Hz to %i Hz" % (old_rate, new_rate)
	new_length = (len(a) * float(new_rate)) / old_rate
	# Prevent some overflow issues.
	a = list(a) + [a[-1]]
	b = []
	for i in xrange(int(new_length)):
		index = i * (old_rate / float(new_rate))
		l = int(index)
		coef = index - l
		b.append((1 - coef) * a[l] + coef * a[l+1])
	return b

if len(sys.argv) != 3:
	print "Usage: resample.py <new sample rate> <input file.wav>"
	print "Input must be mono, 16-bit signed litle endian audio."
	print "Will produce an output file produced by linear interpolation."
	print "Example:"
	print "  ffmpeg -i input.mp3 -ac 1 test_case.wav"
	print "  python resample.py 16000 test_case.wav"
	print "  ffmpeg -i input.mp3 -ac 1 -ar 16000 good_16000.wav"
	print "Now good_16000.wav has ffmpeg's best interpolation job,"
	print "while lerp_16000.wav has our crummy lerp filter used."
	print "Do A/B comparison, to see if you can tell the difference."
	exit()

w = wave.open(sys.argv[2])
assert w.getnchannels() == 1, "Input must be mono!"
assert w.getsampwidth() == 2, "Input must be 16-bit per sample!"
length = w.getnframes()
frames = w.readframes(length)
data = struct.unpack("<%ih" % length, frames)
w.close()

target_rate = int(sys.argv[1])

b = resample(data, w.getframerate(), target_rate)

w = wave.open("lerp_%i.wav" % target_rate, "w")
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(target_rate)
w.writeframes(struct.pack("<%ih" % len(b), *tuple(map(int, b))))
w.close()

