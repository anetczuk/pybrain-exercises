#!/bin/bash

set -eu


SCRIPT_DIR=$(cd ${0%/*} && pwd -P)


cd "$SCRIPT_DIR"


## install requirements
pip2 install -r $SCRIPT_DIR/requirements.txt


SRC_ARCHIVE=pybrain-master.zip

## if [ "$#" -ge 1 ]; then
##     SRC_ARCHIVE=$1
## fi


TMP_ZIP=$(mktemp -t "$SRC_ARCHIVE.XXXXXX.zip")


wget -c https://github.com/anetczuk/pybrain/archive/refs/heads/master.zip -O "$TMP_ZIP" 

## wget -c https://github.com/pybrain/pybrain/archive/refs/heads/master.zip -O "$TMP_ZIP" 


SRC_DIR=$(basename -s .zip $SRC_ARCHIVE)
SRC_ROOT_PATH="${SRC_DIR}-src.XXXXX"
# SRC_PATH="$SRC_ROOT_PATH/${SRC_DIR}"


unzip_src() {
    echo "extracting and overwriting sources"

    ## rm -rf "$SRC_PATH"
    ## mkdir -p "$SRC_PATH"
    
    local WORK_DIR=$(mktemp -d -t "${SRC_ROOT_PATH}")
    SRC_PATH="$WORK_DIR/${SRC_DIR}"

    unzip -o "$TMP_ZIP" -d "$WORK_DIR"
    ## unzip -o "$SCRIPT_DIR/$SRC_ARCHIVE" -d "$WORK_DIR"
}

unzip_src


cd $SRC_PATH

python2 setup.py install


rm "$TMP_ZIP"


echo -e "\npybrain installed"
