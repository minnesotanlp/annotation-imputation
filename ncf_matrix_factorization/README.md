# What is this?
This is the folder containing the code for doing NCF Matrix Factorization.

# How can I use it?
To run it, use `run.sh`. Usually this is done via `rm nohup.out ; nohup bash run.sh & python3 ../utilities/display_realtime_file.py nohup.out`.

In order to use this, you need to have a `.npy` file in the `NpyAnnotationsFormat` format, as specified in `../kernel_matrix_factorization/dataset_formats.py`.

This script will take out the user-specified fold(s) from the dataset, and impute the rest of the matrix.

For other scripts to use these imputed files, you will likely need to move them to the dataset folder by hand. I believe the default location for them in the scripts is to put them directly into the corresponding `cleaned_folds` folder. For example, once moved, there should be files such as `datasets/cleaned_folds/SChem/SChem_0_ncf_imputation_0.npy`.

Please contact the authors if you would like the original imputed datasets for more exact paper replication (rather than generating your own).