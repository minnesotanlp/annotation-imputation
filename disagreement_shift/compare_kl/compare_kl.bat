@echo off
setlocal enabledelayedexpansion

@REM set "datasets=Fixed2xSBIC Sentiment SChem SChem5Labels"
set "datasets=ghc"
set "output_folder=../../datasets/final_imputation_results/distribution_analysis/kl_comparisons"

for %%d in (%datasets%) do (
    set "kl1=../../datasets/final_imputation_results/distribution_analysis/ncf/%%d_kl.npy"
    set "kl2=../../datasets/final_imputation_results/distribution_analysis/kernel/%%d_kl.npy"
    set "kl3=../../datasets/final_imputation_results/distribution_analysis/multitask/%%d_kl.npy"
    
    set "output_file=%output_folder%/%%d_comparison12.json"
    python compare_kl.py --kl1 "!kl1!" --kl2 "!kl2!" --output_file "!output_file!" --name1 "NCF" --name2 "Kernel"

    set "output_file=%output_folder%/%%d_comparison23.json"
    python compare_kl.py --kl1 "!kl2!" --kl2 "!kl3!" --output_file "!output_file!" --name1 "Kernel" --name2 "Multitask"

    set "output_file=%output_folder%/%%d_comparison13.json"
    python compare_kl.py --kl1 "!kl1!" --kl2 "!kl3!" --output_file "!output_file!" --name1 "NCF" --name2 "Multitask"
)