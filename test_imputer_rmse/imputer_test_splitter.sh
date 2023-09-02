#!/bin/env bash

# Check if python3 is available
if command -v python3 &>/dev/null; then
    python_command="python3"
# Check if python is available
elif command -v python &>/dev/null; then
    python_command="python"
else
    echo "Error: Neither python nor python3 is installed on this system."
    exit 1
fi

# All the datasets
# datasets=(2xSBIC politeness Sentiment ghc SChem SChem5Labels)
# Just a single dataset
datasets=(SChem)

for d in "${datasets[@]}"; do
    echo "Splitting ${d} into train and test sets..."
    $python_command imputer_test_splitter.py \
    --annotations ../datasets/cleaned/${d}_annotations.npy \
    --texts ../datasets/cleaned/${d}_texts.npy \
    --train_output ../datasets/rmse_test/${d}_train.csv \
    --test_output ../datasets/rmse_test/${d}_test.csv
done