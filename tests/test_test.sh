#!/bin/bash

echo "Test the agent testing"

PROJECTDIR=$(dirname $(dirname $0))

MAIN_LOC="${PROJECTDIR}/src/main.py"

TEST_MODEL_DIR="${PROJECTDIR}/rsc/models"

python "${MAIN_LOC}" --test --model "${TEST_MODEL_DIR}" --episodes 5

echo "Test end"