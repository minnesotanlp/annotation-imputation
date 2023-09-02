# What does this do?
This computes the RMSE score for median, kernel, and NCF imputation on different datasets.
It's currently the best test we have and is more reliable than their training RMSE scores.
It feeds the model data and cuts out particular examples.
It then measures the RMSE score on examples that the imputation method was never given.

Since NCF doesn't impute the entire matrix, NCF will be judged on fewer examples.

# How to use this
1. Put your dataset through the `imputer_test_splitter`. That script will convert your npy files for the dataset into train and test _matrix_ format (not _npy_ format) datasets. (Matrix format datasets are loaded with Pandas and saved as CSV files. See `../kernel_matrix_factorization/dataset_formats.py` for details)
2. Put these matrix format datasets through the `test_imputers` script, which should be fairly easily modifiable to add new imputers, in case you have a new imputation algorithm you'd like to test.


Then, you should get a folder with results and another folder with extra data. (The default is to put results into the `datasets` folder.) These results will contain the values to recreate the table, along with additional information about the exact predictions and statistics about those predictions for each method.

You can also then run the `display_graphs/display_graphs.py` script on the results in order to visualize them (visualization not included in the paper.)