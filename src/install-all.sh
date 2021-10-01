#!/bin/bash

set -eu

## works both under bash and sh
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")


pip2 install scipy==0.19.1
pip2 install numpy==1.8.2
pip2 install matplotlib==2.2.5

## for ode environments
pip2 install pyode

## for ode environemnts (for viewing)
pip2 install pyopengl
pip2 install pillow

## does not have ode envoronment model files
# pip2 install pybrain

## install local pybrain library
#$SCRIPT_DIR/install-py2.sh pybrain-0.3.3.zip

$SCRIPT_DIR/install-pybrain.sh


### for ode environments
### pyode worjs only under python2
## pip3 install pyode
    
## pybrain-master.zip is not compatible with python3


## install requirements
pip2 install -r $SCRIPT_DIR/requirements.txt


echo -e "\ninstallation done"
