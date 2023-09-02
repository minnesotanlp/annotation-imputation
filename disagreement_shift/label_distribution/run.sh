#!/bin/bash

# Call the check_python.sh script
source ../check_python.sh # set $python_command

# Now you can use the $python_command variable in this script
echo "Using Python command: $python_command"

datasets=("Sentiment" "ghc", "SChem", "SChem5Labels", "politeness")
ncf_folder="../datasets/disagreement_datasets/ncf"
kernel_folder="../datasets/disagreement_datasets/kernel"

folder="$kernel_folder"

for d in "${datasets[@]}"; do
    echo "Processing dataset: $d"

    # For Kernel
    $python_command main.py --orig "$folder/${d}_annotations_cleaned.npy" --imputed "$folder/${d}_annotation_kernel_preds.npy" --orig_distribution_output "$folder/${d}_orig_distribution.npy" --imputed_distribution_output "$folder/${d}_imputed_distribution.npy" --kl_output "$folder/${d}_kl.npy"

    # See run.bat for NCF

    echo "Done with dataset: $d"
done