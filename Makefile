
CFLAGS=-O3 -ffast-math -Wall -g -lm -I.

APP_SOURCES := $(wildcard apps/*.c)
APPS := $(patsubst %.c,%.app,$(APP_SOURCES))

all: $(APPS)

%.app: %.c tinysr.o
	gcc $(CFLAGS) -o $@ $< tinysr.o

.PHONY: clean
clean:
	rm -f *.o

