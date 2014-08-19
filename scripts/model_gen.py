#! /usr/bin/python

import os, sys

class Utterance:
	def __init__(self, path):
		self.fvs = []
		with open(path) as f:
			for line in f.readlines():
				line = line.strip()
				if not line: continue
				self.fvs.append(map(float, line.split(","))[1:])

class GaussianModel:
	def __init__(self, samples):
		self.mean = sum(samples)/len(samples)
		

class Model:
	def __init__(self, stacks):
		self.stacks = stacks

utters = []

for path in os.listdir(sys.argv[1]):
	utters.append(Utterance(os.path.join(sys.argv[1], path)) )

for u in utters:
	print u.fvs

