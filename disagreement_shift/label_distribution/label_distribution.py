import numpy as np
from typing import List

def label_distribution(labels: List[int], arr : np.ndarray) -> np.ndarray:
    """
    Compute the label distribution of each row in the given array.

    :param labels: list of k unique labels
    :param arr: numpy array of shape (n,m) consisting of n examples with m labels or -1 (missing labels) each
    :return: numpy array of shape (n,k) representing the label distribution of each row
    """
    k = len(labels)
    n, m = arr.shape
    dist = np.zeros((n, k))

    for i in range(n):
        row = arr[i]
        valid_labels = row[row != -1]
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            j = labels.index(label)
            dist[i,j] = count / len(valid_labels)

    return dist

def single_row_label_distribution(labels: List[int], row: np.ndarray) -> np.ndarray:
    """
    Compute the label distribution of a single row.

    :param labels: list of k unique labels
    :param row: numpy array of shape (m,) consisting of labels or -1
    :return: numpy array of shape (k,) representing the label distribution of the row
    """
    return label_distribution(labels, row.reshape(1, -1))[0]

if __name__ == "__main__":
    # arr = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                 [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    #                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    arr2 = np.array([1, 1, 2, 3])
    # labels = sorted(list(np.unique(arr[arr != -1])))
    labels = sorted(list(np.unique(arr2)))
    # print(label_distribution(labels, arr))
    print(label_distribution(labels, arr2.reshape(1, -1))[0])