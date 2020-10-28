#!/bin/bash

all=(bike_translation_slow_0_10 bike_translation_slow_0_30 bike_translation_slow_30_60 fox_head_short_0_20 fox_head_short_0_45 house_free_motion_0_10 house_free_motion_30_60 ironman_translation_fast_0_10)

for i in ${all[@]}; do
    echo "$i"
    ./camtrack/cmptrack.py dataset/$(echo $i | sed 's/_[0-9]\+_[0-9]\+$//')/ground_truth.yml output2/$i/track.yml
    echo ""
done
