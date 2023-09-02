#!/bin/bash

# Uses Multitask as an (initial) imputer

# You probably want to run this with something like rm nohup.out ; nohup bash impute.sh & python3 ../utilities/display_realtime_file.py nohup.out
# Do make sure you are logged into Weights and Biases first, or change the logger to use Tensorboard (untested)

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

# For full run, do all datasets:
# datasets=("SChem" "SChem5Labels" "politeness" "ghc" "Fixed2xSBIC" "Sentiment")
# Or, for a quick test, do just one dataset:
datasets=("SChem")
folds=("-1" "0" "1" "2" "3" "4")
# folds=("0")

total_datasets=${#datasets[@]}
total_folds=${#folds[@]}

start_time=$(date +%s)
fold_count=0
# any warnings about creating a validation dataset with all missing entries can be ignored.
# this occurs because we are using all of the data for training, so the validation dataset is created but is empty.
for dataset_index in $(seq 0 $((total_datasets - 1))); do
    dataset=${datasets[$dataset_index]}
    dataset_percentage=$((100 * dataset_index / total_datasets))

    for fold_index in $(seq 0 $((total_folds - 1))); do
        fold=${folds[$fold_index]}
        fold_percentage=$((100 * fold_index / total_folds))

        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        elapsed_time_str=$(printf "%02d:%02d:%02d" $((elapsed_time/3600)) $((elapsed_time%3600/60)) $((elapsed_time%60)))

        fold_count=$((fold_count + 1))
        average_time_per_fold=$((elapsed_time / fold_count))
        average_time_per_fold_str=$(printf "%02d:%02d:%02d" $((average_time_per_fold/3600)) $((average_time_per_fold%3600/60)) $((average_time_per_fold%60)))

        echo "[$elapsed_time_str] Dataset: $dataset ($dataset_percentage% complete) | Fold: $fold ($fold_percentage% complete) | Avg time per fold: $average_time_per_fold_str"

        $python_command impute.py --annotations ../datasets/cleaned/${dataset}_annotations.npy --texts ../datasets/cleaned/${dataset}_texts.npy --output_annotations ../datasets/multitask_imputation_results/${dataset}_${fold}_multitask_annotations.npy --output_texts ../datasets/multitask_imputation_results/${dataset}_${fold}_multitask_texts.npy --output_json ../datasets/multitask_imputation_results/${dataset}_${fold}_multitask.json --fold $fold --epochs 10 --train_split 1 --logger_type wandb --logger_params "{\"project\": \"nlperspectives_replication\", \"name\": \"${dataset}_${fold}_multitask\", \"save_json\": true}"
    done
done