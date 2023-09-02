#!/bin/bash

# Trains the multitask model on a given dataset
# Do make sure you are logged into Weights and Biases first, or change the logger to use Tensorboard (untested)

# SChem5Labels is left out because we don't have NCF imputations for some of the folds
# datasets=("SChem" "politeness" "ghc" "Fixed2xSBIC" "Sentiment")
# accidentally put in comma typo, running the left-out datasets
datasets=("ghc" "politeness" "Fixed2xSBIC")
folds=(0 1 2 3 4)
original_or_imputed_list=("original" "imputed")
# original_or_imputed_list=("imputed")
name="full_run1"

model_type="multi"
logger_type="wandb"
wandb_project="final_multi_anno_runs"
epochs=10

for dataset in "${datasets[@]}"; do
  for fold in "${folds[@]}"; do
    for original_or_imputed in "${original_or_imputed_list[@]}"; do
      # if original or imputed
      if [ "${original_or_imputed}" == "original" ]; then
        # where to find original annotations
        train_annotations="../datasets/cleaned_folds/${dataset}/${dataset}_train_${fold}_annotations.npy"
      else
        # where to find imputed annotations
        # multitask imputation results
        # train_annotations="../datasets/multitask_imputation_results/${dataset}_${fold}_multitask_annotations.npy"

        # ncf imputation results
        train_annotations="../datasets/cleaned_folds/${dataset}/${dataset}_${fold}_ncf_imputation_${fold}.npy"
      fi

      train_texts="../datasets/cleaned_folds/${dataset}/${dataset}_train_${fold}_texts.npy"
      val_annotations="../datasets/cleaned_folds/${dataset}/${dataset}_val_${fold}_annotations.npy"
      val_texts="../datasets/cleaned_folds/${dataset}/${dataset}_val_${fold}_texts.npy"

      echo "Running for dataset: ${dataset}, fold: ${fold}, imputed?: ${original_or_imputed}"
      python main.py "${train_annotations}" "${train_texts}" "${val_annotations}" "${val_texts}" --model_type "${model_type}" --logger_type "${logger_type}" --wandb_project "${wandb_project}" --wandb_run_name "${dataset}_${original_or_imputed}_${fold}_${name}" --epochs "${epochs}" --save_logs --save_logs_where "./logs/${dataset}_${original_or_imputed}_${fold}_${name}.json"
      done
  done
done