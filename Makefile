
CFLAGS=-O3 -ffast-math -Wall -g -lm

all: test_tinysr

test_tinysr: test_tinysr.o tinysr.o
	gcc $(CFLAGS) -o $@ $< tinysr.o

.PHONY: clean
clean:
	rm -f test_tinysr *.o

