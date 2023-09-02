'''
Test to see if two .csv files are the same
'''

import pandas as pd

# datasets = ["SChem", "SChem5Labels", "politeness", "ghc", "Sentiment"]
datasets = ["SChem"]
for dataset in datasets:
    print(dataset)
    # file1 = f"../datasets/cleaned/{dataset}_annotations.npy"
    # file2 = f"../datasets/windows_cleaned/{dataset}_annotations.npy"
    file1 = f"../datasets/imputer_test5/{dataset}_train.csv"
    file2 = f"../datasets/imputer_test6/{dataset}_train.csv"
    arr1 = pd.read_csv(file1)
    arr2 = pd.read_csv(file2)
    print(arr1.shape)
    print(arr2.shape)
    # determine if the dataframes are equal
    print(arr1.equals(arr2))