import numpy as np

def generate_permutation(array, seed=None):
    if seed is not None:
        np.random.seed(seed)
    row_permutation = np.random.permutation(array.shape[0])
    col_permutation = np.random.permutation(array.shape[1])
    return row_permutation, col_permutation

def apply_permutation(array, permutation):
    row_permutation, col_permutation = permutation
    return array[np.ix_(row_permutation, col_permutation)]

def undo_permutation(array, permutation):
    row_permutation, col_permutation = permutation
    undo_row_permutation = np.argsort(row_permutation)
    undo_col_permutation = np.argsort(col_permutation)
    return array[np.ix_(undo_row_permutation, undo_col_permutation)]

if __name__ == "__main__":
    # Example usage
    array = np.array([[20, 2, 3], [4, 50, 6], [7, 8, 9]])
    print("Original array:")
    print(array)

    permutation = generate_permutation(array, seed=1)
    print("\nGenerated permutation:")
    print(permutation)

    permuted_array = apply_permutation(array, permutation)
    print("\nPermuted array:")
    print(permuted_array)

    unpermuted_array = undo_permutation(permuted_array, permutation)
    print("\nUnpermuted array:")
    print(unpermuted_array)