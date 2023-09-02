'''
Simple validator just to make sure every row follows the right format
'''

import pandas as pd

# imports from Kernel Matrix Factorization
import sys
sys.path.append("../kernel_matrix_factorization")
from dataset_formats import MatrixDataFrame
from dataset_columns import SENTENCE_ID_COL, SENTENCE_COL

def assert_valid_matrix_data_frame(df: MatrixDataFrame, unique=True) -> bool:
    '''
    Make sure that the columns are correct and that every row has a sentence and a sentence id
    If unique is True, make sure that every sentence id and sentence is unique
    '''
    # Make sure that the columns are correct
    assert df.columns[0] == SENTENCE_ID_COL
    assert df.columns[1] == SENTENCE_COL
    assert len(df.columns) > 2

    # Make sure that every row has a sentence and a sentence id
    assert df[SENTENCE_ID_COL].notnull().all()
    assert df[SENTENCE_COL].notnull().all()

    if unique:
        # Make sure that every sentence id and sentence is unique
        assert df[SENTENCE_ID_COL].is_unique, f"Duplicate rows found: {df[df.duplicated(subset=[SENTENCE_ID_COL], keep=False)][SENTENCE_ID_COL]}"
        # assert df[SENTENCE_COL].is_unique, f"Duplicate rows found: {df[df.duplicated(subset=[SENTENCE_COL], keep=False)][SENTENCE_COL]}"

    return True