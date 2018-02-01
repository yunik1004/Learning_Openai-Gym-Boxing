#!/bin/bash

echo "Test the agent testing"

PROJECTDIR=$(dirname $(dirname $0))

MAIN_LOC="${PROJECTDIR}/src/main.py"

TEST_MODEL_DIR="${PROJECTDIR}/dataset/models"

python "${MAIN_LOC}" --test --model "${TEST_MODEL_DIR}/sample/model.ckpt" --episodes 2

echo "Test end"