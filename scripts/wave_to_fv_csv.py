#! /usr/bin/python
"""
Script to produce feature vector CSVs from a wave file.
"""

import wave, sys, subprocess, struct
from os.path import join, dirname, abspath

if len(sys.argv) != 3:
	print "Usage: %s <input.wav> <output.csv>" % sys.argv[0]
	print "Converts a wave file to a feature vector CSV file."
	print "Make sure you've compiled compute_fv.app first."
	exit(1)

w = wave.open(sys.argv[1])
assert w.getsampwidth() == 2, "Must be 16-bit wave file!"
assert w.getnchannels() in (1, 2), "Must be mono or stereo!"

downmix = w.getnchannels() == 2
if downmix:
	print "Down-mixing stereo to mono."

print "Gathering up audio."
audio = []
for i in xrange(w.getnframes()):
	frame = w.readframes(1)
	# Downmix the audio if necessary.
	if downmix:
		left, right = struct.unpack("<2h", frame)
		frame = struct.pack("<h", (left+right)/2)
	audio.append(frame)
audio = "".join(audio)

print "Passing to subprocess."
compute_fv_path = join(dirname(dirname(abspath(sys.argv[0]))), "apps", "compute_fv.app")
proc = subprocess.Popen([compute_fv_path, str(w.getframerate()), "/dev/stdin"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
stdout, stderr = proc.communicate(audio)

print "Done processing, writing out CSV."
with open(sys.argv[2], "w") as f:
	f.write(stdout)

