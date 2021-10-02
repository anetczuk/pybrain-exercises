#!/bin/bash

set -eu

## ‘**’ used in a filename expansion context will match all files and zero or more directories and subdirectories
shopt -s globstar


## works both under bash and sh
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")


big_suffix="-big.png"
small_suffix="-small.png"

for filename in $SCRIPT_DIR/**; do
    if [[ $filename != *"${big_suffix}"* ]]; then
        continue
    fi
    small_name=${filename/$big_suffix/$small_suffix}
    echo "converting: $filename -> $small_name"
    convert $filename -resize 240 $small_name
    #convert $filename -resize 200x100 $small_name
done
