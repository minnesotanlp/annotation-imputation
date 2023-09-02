@echo off
setlocal enabledelayedexpansion

echo Using Python command: python

@REM set datasets=SChem SChem5Labels politeness Fixed2xSBIC Sentiment ghc
@REM set datasets=SChem
set datasets=ghc

set folder=..\..\datasets\final_imputation_results

for %%d in (%datasets%) do (
    echo Processing dataset: %%d
    set original=!folder!\original\%%d_annotations.npy
    set originalOutput=!folder!\distribution_analysis\original\%%d_orig_distribution.npy

    @REM For Kernel
    echo Working on Kernel
    set smallfolder=!folder!\kernel_imputation_results
    set outputfolder=!folder!\distribution_analysis\kernel
    set outputFile=!outputfolder!\%%d_imputed_distribution.npy
    set klOutput=!outputfolder!\%%d_kl.npy
    echo Running the following command:
    echo python main.py --orig !original! --imputed !smallfolder!\Fold_-1\%%d_annotations_kernel_preds.npy --orig_distribution_output !originalOutput! --imputed_distribution_output !outputFile! --kl_output !klOutput!
    echo Now running...
    python main.py --orig !original! --imputed !smallfolder!\Fold_-1\%%d_annotations_kernel_preds.npy --orig_distribution_output !originalOutput! --imputed_distribution_output !outputFile! --kl_output !klOutput!

    @REM For NCF
    echo Working on NCF
    set smallfolder=!folder!\ncf_imputation_results
    set outputfolder=!folder!\distribution_analysis\ncf
    set outputFile=!outputfolder!\%%d_imputed_distribution.npy
    set klOutput=!outputfolder!\%%d_kl.npy
    echo Running the following command:
    echo python main.py --orig !original! --imputed !smallfolder!\%%d_-1_ncf_imputation_-1.npy --orig_distribution_output !originalOutput! --imputed_distribution_output !outputFile! --kl_output !klOutput!
    echo Now running...
    python main.py --orig !original! --imputed !smallfolder!\%%d_-1_ncf_imputation_-1.npy --orig_distribution_output !originalOutput! --imputed_distribution_output !outputFile! --kl_output !klOutput!

    @REM For Multitask
    echo Working on Multitask
    set smallfolder=!folder!\multitask_imputation_results
    set outputfolder=!folder!\distribution_analysis\multitask
    set outputFile=!outputfolder!\%%d_imputed_distribution.npy
    set klOutput=!outputfolder!\%%d_kl.npy
    echo Running the following command:
    echo python main.py --orig !original! --imputed !smallfolder!\%%d_-1_multitask_annotations.npy --orig_distribution_output !originalOutput! --imputed_distribution_output !outputFile! --kl_output !klOutput!
    echo Now running...
    python main.py --orig !original! --imputed !smallfolder!\%%d_-1_multitask_annotations.npy --orig_distribution_output !originalOutput! --imputed_distribution_output !outputFile! --kl_output !klOutput!

    echo Done with dataset: %%d
)