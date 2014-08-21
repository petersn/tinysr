#! /usr/bin/python
"""
Playing with multivariate Gaussians.
"""

import numpy, math, random
from matplotlib import pyplot as plt

dims = 5
sample_count = 200
uniform_noise = 1.0

# First, let's generate some multivariate data.
# We need a symmetric positive definite matrix to be the covariance matrix.
while True:
	linop = (numpy.random.random((dims, dims)) * 2 - numpy.ones((dims, dims))) * 8
	linop += linop.transpose()
	if all(i > 0 for i in numpy.linalg.eig(linop)[0]):
		break

def draw():
	return numpy.random.multivariate_normal(numpy.zeros((dims,)), linop) + random.uniform(-uniform_noise, uniform_noise)
#	v = numpy.array([numpy.random.normal() for _ in xrange(dims)])
#	return linop.dot(v)

data = [draw() for _ in xrange(sample_count)]

print data[0]

# Make the data meanless.
mean = sum(data) / len(data)
print "Mean:", mean
data = [x - mean for x in data]

xs = [x[0] for x in data]
ys = [x[1] for x in data]

#plt.scatter(xs, ys)
#plt.show()

# Set up a log likelihood validator function.

def log_likelihood(covariance):
	# Regularization term, of 1/sqrt(|covariance|), in likelihood space.
	ll = -0.5 * math.log(numpy.linalg.det(covariance)) * len(data)
	# Main contribution to log likelihood:
	cov_inv = numpy.linalg.inv(covariance)
	for x in data:
		ll -= 0.5 * x.dot(cov_inv.dot(x))
	return ll

# Now, we do maximum likelihood estimation, to get an estimate of the Gaussian's covariance matrix.

cov = sum(numpy.outer(x, x) for x in data) / len(data)

print "Log likelihood of result:", log_likelihood(cov)

print "Linear operator:"
print linop

print "ML Covariance:"
print cov

eig_vals, eig_vecs = numpy.linalg.eig(cov)

print "Eigen decomposition:"
for i in xrange(dims):
	print "Eigenvalue:", eig_vals[i]
	print "Eigenvector:", eig_vecs[:,i]

