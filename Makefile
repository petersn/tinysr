
CFLAGS=-O3 -ffast-math -Wall -g -lm -I.

APP_SOURCES := $(wildcard apps/*.c)
APPS := $(patsubst %.c,%,$(APP_SOURCES))

all: $(APPS)

apps/%: %.c tinysr.o
	gcc -o $@ $< tinysr.o $(CFLAGS)

.PHONY: clean
clean:
	rm -f *.o

