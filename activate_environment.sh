#!/bin/bash

CURRENT_DIR=$(pwd)

cd ..
source mip-venv-pw/bin/activate
cd "$CURRENT_DIR"

echo "Virtual environment activated and returned to $CURRENT_DIR"
