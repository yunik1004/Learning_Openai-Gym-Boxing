#!/bin/bash

echo "Test the agent training"

PROJECTDIR=$(dirname $(dirname $0))

MAIN_LOC="${PROJECTDIR}/src/main.py"

python "${MAIN_LOC}" --train --episodes 10

echo "Test end"