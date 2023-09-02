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

# All datasets
# datasets=(SChem politeness SChem5Labels Fixed2xSBIC Sentiment ghc)
# Just one dataset
datasets=(SChem)

n_folds=5
folds=("-1" "0" "1" "2" "3" "4")
# folds=("-1") # to impute an entire dataset

for dataset in ${datasets[@]}
do
    for fold in ${folds[@]}
    do
        echo "Imputing ${dataset} fold ${fold}..."

        # # quick test (limited grid search)
        # $python_command main.py \
        # --dataset ${dataset} \
        # --fold ${fold} \
        # --n_folds ${n_folds} \
        # --input_npy_annotations_file ../datasets/cleaned/${dataset}_annotations.npy \
        # --input_npy_texts_file ../datasets/cleaned/${dataset}_texts.npy \
        # --output_npy_annotations_file ../datasets/kernel_test/Fold_${fold}/${dataset}_annotations_kernel_preds.npy \
        # --output_json_file ../datasets/kernel_test/run_data/${dataset}_fold${fold}_annotation_kernel_preds.json \
        # --log_file ../datasets/kernel_test/run_data/${dataset}_fold${fold}_kernel_matrix_factorization_imputation_log.txt \
        # --n_factors 4 1 \
        # --n_epochs 2 1 \
        # --kernels linear rbf sigmoid \
        # --gammas auto \
        # --regs 0.01 \
        # --lrs 0.01 0.001 \
        # --init_means 0 \
        # --init_sds 0.1 \
        # --seeds 85 \
        # --allow_duplicates

        # full kernel imputation command (full grid search)
        $python_command main.py \
        --dataset ${dataset} \
        --fold ${fold} \
        --n_folds ${n_folds} \
        --input_npy_annotations_file ../datasets/cleaned/${dataset}_annotations.npy \
        --input_npy_texts_file ../datasets/cleaned/${dataset}_texts.npy \
        --output_npy_annotations_file ../datasets/kernel/Fold_${fold}/${dataset}_annotations_kernel_preds.npy \
        --output_json_file ../datasets/kernel/run_data/${dataset}_fold${fold}_annotation_kernel_preds.json \
        --log_file ../datasets/kernel/run_data/${dataset}_fold${fold}_kernel_matrix_factorization_imputation_log.txt \
        --n_factors 32 16 8 4 2 1 \
        --n_epochs 256 128 64 32 16 8 4 2 1 \
        --kernels linear rbf sigmoid \
        --gammas auto \
        --regs 0.1 0.01 0.001 \
        --lrs 0.01 0.001 0.0001 \
        --init_means 0 \
        --init_sds 0.1 \
        --seeds 42 85 \
        --allow_duplicates
        
        echo "Imputation of ${dataset} fold ${fold} done"
    done
done

echo "All imputations complete."