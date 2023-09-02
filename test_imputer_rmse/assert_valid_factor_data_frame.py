'''
Check to make sure that a factor data frame is in the right format
'''

import pandas as pd

# imports from `kernel_matrix_factorization`
import sys
sys.path.append("../kernel_matrix_factorization")
from dataset_formats import FactorDataFrame
from dataset_columns import FACTOR_FORMAT_COLS, SENTENCE_ID_COL, USER_ID_COL

def assert_valid_factor_data_frame(df: FactorDataFrame) -> bool:
    # Make sure all the columns are there - order doesn't matter
    assert set(df.columns) == set(FACTOR_FORMAT_COLS), f"Expected columns {FACTOR_FORMAT_COLS}, got {list(df.columns)}"
    # Make sure there are no nans
    assert (df.notnull().all().all()), "DataFrame contains NaN values"
    # Make sure that there are no rows with the same sentence id and user id
    assert not df.duplicated(subset=[SENTENCE_ID_COL, USER_ID_COL]).any(), "DataFrame contains duplicate rows"

    return True

