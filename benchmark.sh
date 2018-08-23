#!/bin/bash

USERS=2e6
ITEMS=2e5
EXAMPLES=1e7
DESTFILE="/Users/aschioppa/speedTFData.tfr"

time python tests/generateUserItemExas.py \
     --users $USERS --items $ITEMS \
     --examples $EXAMPLES \
     --dest_file $DESTFILE

echo -e '\nData Size'
echo $(du -h $DESTFILE)
