#!/bin/bash

USERS=2e6
ITEMS=2e5
DESTFILE="/Users/aschioppa/speedTFData.tfr"
BATCH=64
LATENTFACTORS=64
NUMNEGS=32

python tests/shardedNegBPRTest.py --users $USERS \
        --items $ITEMS \
        --num_latent $LATENTFACTORS \
        --batch_size $BATCH \
        --source_file $DESTFILE \
	--num_negs $NUMNEGS \
	--examples 1e7
