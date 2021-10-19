#!/bin/bash

set -eu


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"


VENV_SUBDIR=""
if [ "$#" -ge 1 ]; then
    VENV_SUBDIR=$1
fi

VENV_DIR=$(realpath "$SCRIPT_DIR/../venv/$VENV_SUBDIR")


### if directory exists then prompt to delete

if [ -d "$VENV_DIR" ]; then
    read -p "Directory [$VENV_DIR] exists. Do You want to remove it (y/n)? " -n 1 -r
    echo    # (optional) move to a new line
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Given target directory [$VENV_DIR] exists, remove it and restart the script"
        exit 1
    fi
    # do dangerous stuff
    echo "Removing directory [$VENV_DIR]"
    rm -rf "$VENV_DIR"
fi
    

echo "Creating virtual environment in $VENV_DIR"

python2 -m virtualenv $VENV_DIR

# python3 -m venv $VENV_DIR


### creating start script

START_SCRIPT_PATH="$VENV_DIR/start.sh"

START_SCRIPT_CONTENT='#!/bin/bash

set -eu

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

VENV_DIR="$VENV_ROOT_DIR"


START_SCRIPT=
if [ "$#" -ge 1 ]; then
    START_SCRIPT=$1
fi

START_COMMAND=""
if [ ! -z "$START_SCRIPT" ]; then
    START_COMMAND="bash $START_SCRIPT"
fi


### create temporary file
tmpfile=$(mktemp venv.activate.XXXXXX.sh --tmpdir)

### write content to temporary
cat > $tmpfile <<EOL
source $VENV_DIR/bin/activate
if [ \$? -ne 0 ]; then
    echo -e "Unable to activate virtual environment, exiting"
    exit 1
fi

$START_COMMAND

exec </dev/tty 
EOL


echo "Starting virtual env"

bash -i <<< "source $tmpfile"


rm $tmpfile
'


START_SCRIPT_CONTENT="${START_SCRIPT_CONTENT//'$VENV_ROOT_DIR'/$VENV_DIR}"

echo "$START_SCRIPT_CONTENT" > "$START_SCRIPT_PATH"

chmod +x "$START_SCRIPT_PATH"


### install required packages

$START_SCRIPT_PATH "$SCRIPT_DIR/../src/install-all.sh"
