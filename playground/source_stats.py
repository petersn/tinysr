#! /usr/bin/python
"""
Print some statistics about the source code.
"""

comments = 0
source = 0

for path in ("tinysr.c", "tinysr.h"):
	for line in open(path):
		line = line.strip()
		if not line: continue
		if line.startswith("//"):
			comments += 1
		else:
			source += 1

print "Lines of comment:", comments
print "Lines of source: ", source
print "Comments are: %.1f%%" % (comments * 100.0 / (comments + source))

