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
# datasets=("SChem" "SChem5Labels" "politeness" "Fixed2xSBIC" "ghc" "Sentiment")
# Or, for a quick test, do just one dataset:
datasets=("SChem")
# fold -1 means to impute the full dataset (no validation fold)
folds=("-1" "0" "1" "2" "3" "4")

total_datasets=${#datasets[@]}
total_folds=${#folds[@]}

start_time=$(date +%s)
fold_count=0

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

        CUDA_VISIBLE_DEVICES=0 $python_command main_ours_search.py --input_path ../datasets/cleaned/${dataset}_annotations.npy --output_path ${dataset}_${fold}_ncf_imputation --fold $fold --batch_size=256 --epochs=20
    done
done