# multitask

This implements code for the multitask model from Dealing with Disagreements.
This implementation (by London) is primarily different from Jaehyung's (in the main high-level folder) in that it allows you to pick the train and validation file, rather than doing the splitting for you.

## run.sh
This is how you train the multitask model on a given dataset.
It calls `main.py`.

## impute.sh
This is how you use multitask as an imputer for your own datasets.
It uses `impute.py` which uses `multitask_imputer.py`.
Note that for RMSE testing, this script is not used and instead the script can be found in the folder for RMSE testing. (Which will use `multitask_imputer.py`).

Running this will likely require a GPU with ~48GB of RAM. (24GB was too small, even for the small datasets such as `SChem`.)

_(This code was created with assistance from GPT-4)_