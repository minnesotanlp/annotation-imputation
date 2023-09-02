from matrix_to_npy_format import matrix_to_npy_format
from npy_to_matrix_format import npy_to_matrix_format

import numpy as np
import pandas as pd

dataset = np.load('../datasets/cleaned/SChem_annotations.npy', allow_pickle=True)
dataset_texts = np.load('../datasets/cleaned/SChem_texts.npy', allow_pickle=True)

# create fake dataset for testing
# dataset = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# dataset_texts = np.array(['a', 'b', 'c'])

print(f"Shape and dtype of dataset: {dataset.shape}, {dataset.dtype}")
matrix_dataset = npy_to_matrix_format(dataset, dataset_texts)
# print(f"Shape and dtype of matrix_dataset: {matrix_dataset.shape}, {matrix_dataset.dtypes}")
reverted_dataset_annotations, reverted_dataset_texts = matrix_to_npy_format(matrix_dataset)
print(f"Shape and dtype of reverted_dataset: {reverted_dataset_annotations.shape}, {reverted_dataset_annotations.dtype}")

if not np.array_equal(dataset, reverted_dataset_annotations):
    print(dataset[:5])
    print(reverted_dataset_annotations[:5])
    assert False, "The dataset and reverted_dataset are not equal"

if not np.array_equal(dataset_texts, reverted_dataset_texts):
    print(dataset_texts[:5])
    print(reverted_dataset_texts[:5])
    assert False, "The dataset_texts and reverted_dataset_texts are not equal"