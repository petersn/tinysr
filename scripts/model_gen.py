#! /usr/bin/python

import numpy
import os, sys, math, random, struct, time

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
		# covar_inv corresponds to $\Sigma^{-1}$ from the Wikipedia article.
		self.covar_inv = numpy.linalg.inv(self.covariance)
		# This is the log likelihood offset, corresponding to the $\det(\Sigma)^{-1/2}$ factor from Wikipedia.
		self.ll_const = -0.5 * math.log(numpy.linalg.det(self.covariance))

	def ll(self, datum):
		# Compute the log likelihood of a datum matching the model.
		datum = numpy.array(datum) - self.mean
		return self.ll_const - 0.5 * datum.dot(self.covar_inv.dot(datum))

class Model:
	def __init__(self, name, stacks):
		self.name, self.stacks = name, stacks

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

	def write_to_file(self, f):
		# Write the model name out.
		f.write(struct.pack("<I", len(self.name)) + self.name)
		# Write out the offset and slope coefficients.
		f.write(struct.pack("<2f", self.ll_offset, self.ll_slope))
		# Write out how long the template is.
		f.write(struct.pack("<I", len(self.template)))
		for gaussian in self.template:
			# Write out the log likelihood offset for the entry.
			f.write(struct.pack("<f", gaussian.ll_const))
			# Write out the mean.
			f.write(struct.pack("<13f", *gaussian.mean))
			# Write out the inverse covariance matrix.
			for row in gaussian.covar_inv: 
				f.write(struct.pack("<13f", *row))

	@staticmethod
	def read_from_file(self, path):
		# TODO: Finish this.
		with open(path) as f:
			name_length, = struct.unpack("<I", f.read(4))
			name = f.read(name_length)
			template_length, = struct.unpack("<I", f.read(4))
			template = []
			for i in xrange(template_length):
				pass

def read_in_utterance(path):
	fvs = []
	with open(path) as f:
		for line in f.readlines():
			line = line.strip()
			if not line: continue
			# The [1:] skips the log energy entry.
			fvs.append(numpy.array(map(float, line.split(","))[1:]))
	# Make sure each feature vector has 13 coefficients.
	assert len(fvs[0]) == 13
	return fvs

def build_model_from_dir(name, utters_path):
	# Load up all the utterances in the input directory.
	utters = []
	for path in os.listdir(utters_path):
		utters.append(read_in_utterance(os.path.join(utters_path, path)))

	# Start by generating a trivial model from a median length utterance.
	l = list(sorted(utters, key=len))
	candidate = l[len(l)/2]
	print "Total utters:", len(utters), "Lengths:", min(map(len, utters)), len(candidate), max(map(len, utters))
	model = Model(name, [[fv] for fv in candidate])
	model.build_model()

	# Start iterating, making new models.
	old_error, round_num = float("-inf"), 0
	while True:
		round_num += 1
		# Make a model with the same number of stacks as in our previous model.
		new_model = Model(name, [[] for i in xrange(len(model.template))])
		total_fit_error = 0.0
		for u in utters:
			ll, path = model.dynamic_time_warping(u)
			total_fit_error += ll
			# Compose the data according to the path.
			for x, y in path:
				new_model.stacks[y].append(u[x])
		new_model.build_model()
		print "Round %i log likelihood: %f" % (round_num, total_fit_error)
		# Run to convergence.
		if total_fit_error <= old_error: break
		old_error = total_fit_error
		model = new_model

	return utters, model

if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help")):
	print "Usage: model_gen.py dir0 [dir1 ...] output_model"
	print "Each directory is expected to contain utterances in CSV format."
	print "A normalized model will be produced and written to output_model."
	exit(1)

input_paths = sys.argv[1:-1]
output_path = sys.argv[-1]
models = []
start = time.time()

print "=== Building model with %i words." % len(input_paths)
# Build all the base models.
for path in input_paths:
	word_name = os.path.split(path)[1]
	print "== Building %r" % word_name
	utters, model = build_model_from_dir(word_name, path)
	models.append((utters, model, word_name))

print "=== Cross normalizing models."
# Do cross-normalization.
for utters, model, word_name in models:
	# Compute the expected log likelihood across the matching word.
	match_ll = sum(map(model.ll, utters)) / len(utters)
	# Compute the expected log likelihood across wrong words.
	reject_utters = reduce(lambda x, y: x+y, [us for us, m, n in models if m != model])
	reject_ll = sum(map(model.ll, reject_utters)) / len(reject_utters)
	print "== %r" % word_name
	print "match  %f (%i)" % (match_ll, len(utters))
	print "reject %f (%i)" % (reject_ll, len(reject_utters))
	# Compute the coefficients such that (offset + slope * ll) comes out to 0 for matches, and -100 for rejects.
	slope = -100.0 / (reject_ll - match_ll)
	offset = - match_ll * slope
	print "new = %.3f + %.3f * old" % (offset, slope)
	f = lambda x: offset + slope * x
	print "M: %f R: %f" % (f(match_ll), f(reject_ll))
	# Store the coefficients with the model, so they'll get written to the output file.
	model.ll_offset = offset
	model.ll_slope = slope

print "=== Writing output file to: %r" % output_path
with open(output_path, "w") as f:
	for utters, model, word_name in models:
		model.write_to_file(f)

stop = time.time()
print "Done in %f seconds." % (stop - start)

