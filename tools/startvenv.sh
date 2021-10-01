#!/bin/bash

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

VENV_DIR="$SCRIPT_DIR/.."

START_SCRIPT=
if [ "$#" -ge 1 ]; then
    START_SCRIPT=$1
fi


START_COMMAND=""
if [ ! -z "$START_SCRIPT" ]; then
    START_COMMAND="bash $START_SCRIPT"
fi


tmpfile=$(mktemp start.venv.XXXXXX.sh --tmpdir)

cat > $tmpfile <<EOL
source $VENV_DIR/venv/bin/activate
if [ \$? -ne 0 ]; then
    echo -e "Unable to activate virtual environment, exiting"
    exit 1
fi

$START_COMMAND

exec </dev/tty 
EOL


echo "Starting virtual env"

bash -i <<< "source $tmpfile"
