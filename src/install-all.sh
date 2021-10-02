#!/bin/bash

set -eu

## works both under bash and sh
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")


## install local pybrain library
#$SCRIPT_DIR/install-py2.sh pybrain-0.3.3.zip

$SCRIPT_DIR/../lib/pybrain/install.sh


### for ode environments
### pyode worjs only under python2
## pip3 install pyode
    
## pybrain-master.zip is not compatible with python3


## install requirements
pip2 install -r $SCRIPT_DIR/requirements.txt


echo -e "\ninstallation done\n"
