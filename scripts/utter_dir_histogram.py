#! /usr/bin/python
"""
Compute a histogram of data about utterances in a given directory.
"""

import sys, os
from matplotlib import pyplot as plt

if len(sys.argv) != 2:
	print "Usage: utter_dir_histogram.py <utterance directory>"
	exit(1)

lengths = []

for path in os.listdir(sys.argv[1]):
	path = os.path.join(sys.argv[1], path)
	lengths.append(len(open(path).readlines()))

plt.hist(lengths)
plt.show()

