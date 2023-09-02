from typing import Tuple
from dataset_formats import MatrixDataFrame, NpyAnnotationsFormat, NpyTextsFormat
from dataset_columns import SENTENCE_COL
import numpy as np

def matrix_to_npy_format(matrix_df: MatrixDataFrame) -> Tuple[NpyAnnotationsFormat, NpyTextsFormat]:
    # Get the annotations (user ratings)
    annotations = matrix_df.iloc[:, 2:].values.astype(float)
    annotations[np.isnan(annotations)] = -1  # Replace NaN with -1

    # Get the texts (sentences)
    texts = matrix_df[SENTENCE_COL].values.reshape((-1, 1))

    # Convert the texts to the original dtype (str)
    if np.issubdtype(texts.dtype, np.floating):
        texts = texts.astype(str)

    # Return the annotations and texts
    return annotations, texts

if __name__ == '__main__':
    import pandas as pd
    # # Test the function
    # matrix_df = pd.read_csv("../datasets/SChem.csv")
    # annotations, texts = matrix_format_to_npy(matrix_df)
    # print(annotations.shape)
    # print(texts.shape)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='csv file in Matrix format to convert to npy format', required=True)
    args = parser.parse_args()

    matrix_df = pd.read_csv(args.file)
    annotations, texts = matrix_to_npy_format(matrix_df)

    # Save the annotations and texts
    np.save(args.file.replace(".csv", "_annotations.npy"), annotations)
    np.save(args.file.replace(".csv", "_texts.npy"), texts)