@echo off

set DATASETS=politeness politeness_binary Sentiment ghc
set LOAD_LOCATION=..\datasets
set SAVE_LOCATION=..\datasets\cleaned

for %%d in (%DATASETS%) do (
    echo Working on %%d
    python fix_duplicate_texts.py --dataset_name %%d --load_location %LOAD_LOCATION% --save_location %SAVE_LOCATION% --randomize
)