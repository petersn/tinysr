#! /usr/bin/python

import numpy
import os, sys

class Utterance:
	def __init__(self, path):
		self.fvs = []
		with open(path) as f:
			for line in f.readlines():
				line = line.strip()
				if not line: continue
				self.fvs.append(numpy.array(map(float, line.split(","))[1:]))

class MultivariateGaussianModel:
	def __init__(self, vecs):
		# Compute the mean vector.
		self.mean = sum(vecs) / len(vecs)
		# Subtract the mean from the data.
		vecs = [v - self.mean for v in vecs]
		# Compute the covariance matrix.
		# Expectation is maximized by computing this as the expected self outer product over the meanless data.
		# For a derivation of this fact:
		# http://en.wikipedia.org/wiki/Estimation_of_covariance_matrices#Maximum-likelihood_estimation_for_the_multivariate_normal_distribution
		# http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Estimation_of_parameters
		self.covariance = sum(numpy.outer(v, v) for v in vecs) / len(vecs)
		# covar_inv corresponds to $\Sigma^{-1}$ from the Wikipedia article.
		self.covar_inv = numpy.linalg.inv(self.covariance)
		# This is the log likelihood offset, corresponding to the $\det(\Sigma)^{-1/2}$ factor from Wikipedia.
		self.ll_const = -0.5 * math.log(numpy.linalg.det(self.covariance))

	def ll(self, datum):
		# Compute the log likelihood of a datum matching the model.
		datum = numpy.array(datum) - self.mean
		return self.ll_const - 0.5 * datum.dot(self.covar_inv.dot(datum))

class Model:
	def __init__(self, stacks):
		self.stacks = stacks

	def

if len(sys.argv) != 2:
	print "Usage: model_gen.py path/to/words"
	exit(1)

# Load up all the utterances in a directory.
utters = []
for path in os.listdir(sys.argv[1]):
	utters.append(Utterance(os.path.join(sys.argv[1], path)) )

for u in utters:
	print MultivariateGaussianModel(u.fvs)

