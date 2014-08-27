#! /bin/bash
# Simple demo script, so I don't have to keep remembering how to use arecord.

MY_PATH="`dirname \"$0\"`"
arecord -r 16000 -c 1 -f S16_LE | valgrind $MY_PATH/../apps/full_reco.app $MY_PATH/max_model

