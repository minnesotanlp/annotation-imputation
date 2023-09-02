'''
Test to see if two .npy files are the same
'''

import numpy as np
# datasets = ["SChem", "SChem5Labels", "politeness", "ghc", "Sentiment"]
datasets = ["SChem"]
for dataset in datasets:
    print(dataset)
    file1 = f"../datasets/cleaned/{dataset}_annotations.npy"
    file2 = f"../datasets/windows_cleaned/{dataset}_annotations.npy"
    arr1 = np.load(file1, allow_pickle=True)
    arr2 = np.load(file2, allow_pickle=True)
    print(arr1.shape)
    print(arr2.shape)
    print(np.array_equal(arr1, arr2))