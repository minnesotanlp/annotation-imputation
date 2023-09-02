import numpy as np
import pandas as pd
import tempfile
import os
from save_df import save_df
from dataset_formats import MatrixDataFrame, NpyAnnotationsFormat, NpyTextsFormat
from dataset_columns import SENTENCE_ID_COL, SENTENCE_COL, LABEL_COL

def npy_to_matrix_format(npy_annotations: NpyAnnotationsFormat, npy_texts: NpyTextsFormat, allow_duplicates=False) -> MatrixDataFrame:
    # Get the number of samples/users and items/annotations
    # annotations.shape = (400, 100) for SChem = (400 examples, 100 users)
    _num_examples, num_users = npy_annotations.shape

    # Create the index and columns for the dataframe
    columns = [SENTENCE_ID_COL, SENTENCE_COL] + [f"USER{i}" for i in range(num_users)]

    # Create the dataframe
    matrix_df = pd.DataFrame(columns=columns)
    matrix_df[columns[2:]] = npy_annotations

    # set the sentence_id and sentence columns to just be the index of the row they're in
    matrix_df[SENTENCE_ID_COL] = npy_texts.copy()
    matrix_df[SENTENCE_COL] = npy_texts.copy()

    # Replace all the -1's with Nones
    matrix_df = matrix_df.replace(-1, None)

    if not allow_duplicates:
        # check if there are any duplicate sentences (quick T/F check)
        if matrix_df[SENTENCE_COL].duplicated().any():
            raise ValueError("There are duplicate sentences in the dataset. Please remove them before continuing.")
        
    # this somehow changes the typing, but if we save and reload the df, it will be fixed.
    with tempfile.TemporaryDirectory() as tmpdirname:
        # save/load the matrix_df to/from a csv file in the temporary directory
        matrix_path = os.path.join(tmpdirname, "matrix_df.csv")
        save_df(matrix_df, matrix_path)
        matrix_df = pd.read_csv(matrix_path)

    return matrix_df


if __name__ == '__main__':
    # Test the function
    npy_annotations = np.load("../datasets/Sentiment_annotations.npy", allow_pickle=True)
    npy_texts = np.load("../datasets/Sentiment_texts.npy", allow_pickle=True)
    matrix_df = npy_to_matrix_format(npy_annotations, npy_texts)
    print(matrix_df)
    print(matrix_df.columns)
    print(matrix_df.head())