# Instructions on How to Run Kernel Matrix Factorization
Run `run.sh`, which will run Kernel Matrix Factorization and generate imputed datasets for each of the folds and datasets listed in the `run.sh` script. If you have a different location or different schema for your files, edit the locations in `run.sh` or `main.py`.
(If you want a quick run, you can comment out the "full kernel imputation" command and comment in the "quick test" command as indicated by comments in the script.)

For running it in the background / on a remote server, you may want to launch `run.sh` with the command `rm nohup.out ; nohup bash run.sh & python3 ../utilities/display_realtime_file.py nohup.out` which will display the output but run the code in the background and in a way such that even if you close the connection to the server, it will continue to run.

Behind the scenes, the Kernel Matrix Factorization code uses a different file formats (such as Matrix Format or Factor Format as compared to the usual , described in `dataset_formats.py`) converts the .npy format into the "Matrix Format" we were previously using, runs the old code, and then converts the results back.

In addition, there is a component that ensures that the training data always contains at least one annotation for each text and user, so no imputation is done without any data for that example/user.

The code should output files with the suffix `kernel_preds.npy` for the predictions. It should also output files with the suffix `kernel_preds.json` which contain data about the grid search for each dataset.

The data from the `kernel_preds.json` file should be able to be graphed by code like `analysis/graph_mf_grid_search.py`. However, that script uses the old `DatasetIdentifier` class which isn't a part of this project, and hasn't been updated to just take in the location of the JSON. Once it's updated to do that, it should be able to display 2D and 3D graphs that show the relationship between any two variables in the grid search and RMSE score.

If you are looking for the core logic of the kernel matrix factorization imputer, it is in the `kernel_matrix_factorization_imputer.py` file.