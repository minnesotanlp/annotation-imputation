#!/bin/env bash

# you probably want to run this with something like:
# rm -r logs ; rm -r ../datasets/rmse_test/run_data ; rm -r ../datasets/rmse_test/results ; rm nohup.out ; nohup bash test_imputers.sh & python3 ../utilities/display_realtime_file.py ./nohup.out

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
# datasets=(SChem ghc politeness SChem5Labels Fixed2xSBIC Sentiment)
# Just a single dataset
datasets=(SChem)

for d in "${datasets[@]}"; do
    echo "Running tests for $d"
    $python_command test_imputers.py --train ../datasets/rmse_test/${d}_train.csv --test ../datasets/rmse_test/${d}_test.csv --output ../datasets/rmse_test/results/${d}_results.json --data ../datasets/rmse_test/run_data/${d} --python_command $python_command --ncf_main_ours_search ../ncf_matrix_factorization/main_ours_search.py --ncf_batch_size 256 --ncf_epochs 20 --multitask_epochs 5 --multitask_encoder_model 'bert-base-uncased' --multitask_train_split 0.90 --multitask_logger_type 'wandb' --multitask_logger_params "{\"project\": \"nlperspectives_replication\", \"name\": \"multitask_rmse_test_${d}\"}"
done

echo "Done!"