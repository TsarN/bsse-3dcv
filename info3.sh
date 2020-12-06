#!/bin/bash

set -e

all=(bike_translation_slow
    fox_head_full
    fox_head_short
    house_free_motion
    ironman_translation_fast
    room
    soda_free_motion)

for i in ${all[@]}; do
    echo "$i"
    ./camtrack/cmptrack.py dataset/$(echo $i | sed 's/_[0-9]\+_[0-9]\+$//')/ground_truth.yml output3/$i/track.yml
    echo ""
done
