#!/bin/bash

if command -v python3 &>/dev/null; then
    python_command="python3"
elif command -v python &>/dev/null; then
    python_command="python"
else
    echo "Error: Neither python nor python3 is installed on this system."
    exit 1
fi

$python_command main.py \
    --annotations_file ../../datasets/cleaned/SChem_annotations.npy \
    --name "PCA Plot - Original SChem"