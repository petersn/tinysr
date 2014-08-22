#! /usr/bin/python

import numpy
import os, sys, math, random

class MultivariateGaussianModel:
	def __init__(self, vecs):
		# Compute the mean vector.
		self.mean = sum(vecs) / len(vecs)
		# Subtract the mean from the data.
		vecs = [v - self.mean for v in vecs]
		if len(vecs) == 1:
			# If we only have one vector, then the covariance is degenerate, so we just assume unit covariance.
			# This is a hack to get the system started up from a single vector.
			self.covariance = numpy.identity(len(vecs[0]))
		else:
			# Compute the covariance matrix.
			# Expectation is maximized by computing this as the expected self outer product over the meanless data.
			# For a derivation of this fact:
			# http://en.wikipedia.org/wiki/Estimation_of_covariance_matrices#Maximum-likelihood_estimation_for_the_multivariate_normal_distribution
			# http://en.wikipedia.org/wiki/Multivariate_normal_distribution#Estimation_of_parameters
			self.covariance = sum(numpy.outer(v, v) for v in vecs) / len(vecs)
#			self.covariance += numpy.ones(len(vecs[0])) * 1
#			print " ".join("%.2f" % i for i in numpy.linalg.eig(self.covariance)[0])
		# covar_inv corresponds to $\Sigma^{-1}$ from the Wikipedia article.
		self.covar_inv = numpy.linalg.inv(self.covariance)
		# This is the log likelihood offset, corresponding to the $\det(\Sigma)^{-1/2}$ factor from Wikipedia.
#		self.ll_const = -0.5 * math.log(numpy.linalg.det(self.covariance))
		self.ll_const = 0.0

	def ll(self, datum):
		# Compute the log likelihood of a datum matching the model.
		datum = numpy.array(datum) - self.mean
		return self.ll_const - 0.5 * datum.dot(self.covar_inv.dot(datum))

class Model:
	def __init__(self, stacks):
		self.stacks = stacks

	def build_model(self):
		self.template = [MultivariateGaussianModel(stack) for stack in self.stacks]

	def dynamic_time_warping(self, utterance):
		# Match up the utterance via dynamic programming.
		ll = {}
		parent = {}
		def get(xy):
			return ll.get(xy, -float("inf"))
		for x in xrange(len(utterance)):
			for y in xrange(len(self.template)):
				min_predec = max([(x-1, y), (x, y-1), (x-1, y-1)], key=get)
				ll[x, y] = (0.0 if (x, y) == (0, 0) else ll[min_predec]) + self.template[y].ll(utterance[x])
				parent[x, y] = min_predec
		parent[0, 0] = None
		path = [(len(utterance)-1, len(self.template)-1)]
		while path[0] in parent:
			path.insert(0, parent[path[0]])
		return ll[path[-1]], path[1:]

	def ll(self, utterance):
		return self.dynamic_time_warping(utterance)[0]

def read_in_utterance(path):
	fvs = []
	with open(path) as f:
		for line in f.readlines():
			line = line.strip()
			if not line: continue
			fvs.append(numpy.array(map(float, line.split(","))[1:]))
	return fvs

def build_model_from_dir(utters_path):
	# Load up all the utterances in the input directory.
	utters = []
	for path in os.listdir(utters_path):
		utters.append(read_in_utterance(os.path.join(utters_path, path)))

	# Start by generating a trivial model from a median length utterance.
	l = list(sorted(utters, key=len))
	candidate = l[len(l)/2]
	print "Lengths:", min(map(len, utters)), len(candidate), max(map(len, utters))
	model = Model([[fv] for fv in candidate])
	model.build_model()

	# Start iterating, making new models.
	for round_num in xrange(5):
		print "Building round %i model." % round_num
		# Make a model with the same number of stacks as in our previous model.
		new_model = Model([[] for i in xrange(len(model.template))])
		total_fit_error = 0.0
		for u in utters:
			ll, path = model.dynamic_time_warping(u)
			total_fit_error += ll
			# Compose the data according to the path.
			for x, y in path:
				new_model.stacks[y].append(u[x])
		new_model.build_model()
		print "Built %i-model with error: %f" % (len(new_model.template), total_fit_error)
		model = new_model

	return utters, model

print "Building 'one' model."
ones, one_model = build_model_from_dir("../data/one")
print "Building 'two' model."
twos, two_model = build_model_from_dir("../data/two")

print "Beginning cross validation."
for i in xrange(10):
	one, two = random.choice(ones), random.choice(twos)
	print "One-model LLs:", one_model.ll(one), one_model.ll(two)
	print "Two-model LLs:", two_model.ll(one), two_model.ll(two)

