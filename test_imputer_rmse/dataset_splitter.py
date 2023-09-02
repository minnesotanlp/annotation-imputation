from typing import Sequence

import sys
sys.path.append("../kernel_matrix_factorization")
from dataset_formats import MatrixDataFrame, NpyAnnotationsFormat, NpyTextsFormat
from matrix_to_npy_format import matrix_to_npy_format
from npy_to_matrix_format import npy_to_matrix_format

class DatasetSplitter:
    '''Abstract class for a function that has a name and splits a dataset into train/val/test'''
    def __init__(self, name):
        self.name = name
    
    def split(self, df: MatrixDataFrame) -> Sequence[MatrixDataFrame]:
        raise NotImplementedError
    
def split_npy(splitter: DatasetSplitter, annotations: NpyAnnotationsFormat, texts: NpyTextsFormat) -> Sequence[MatrixDataFrame]:
    matrix = npy_to_matrix_format(annotations, texts, allow_duplicates=False)
    return splitter.split(matrix)