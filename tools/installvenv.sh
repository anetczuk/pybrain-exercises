#!/bin/bash

set -eu


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"


ENV_DIR=$SCRIPT_DIR/../venv


if [ -d "$ENV_DIR" ]; then
    read -p "Directory [$ENV_DIR] exists. Do You want to remove it (y/n)? " -n 1 -r
    echo    # (optional) move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Given target directory [$ENV_DIR] exists, remove it and restart the script"
        exit 1
    fi
    # do dangerous stuff
    echo "Removing directory [$ENV_DIR]"
    rm -rf "$ENV_DIR"
fi
    

echo "Creating virtual environment in $ENV_DIR"


python2 -m virtualenv $ENV_DIR

# python3 -m venv $ENV_DIR


$SCRIPT_DIR/startvenv.sh "$SCRIPT_DIR/../src/install-all.sh"

