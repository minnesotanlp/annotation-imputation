import pandas as pd
from enum import Enum
from typing import Tuple
import json
import warnings
import numpy as np

# imports from `kernel_matrix_factorization`
import sys
sys.path.append("../kernel_matrix_factorization")
from imputer import Imputer, JSONstr
from dataset_columns import SENTENCE_COL, SENTENCE_ID_COL
from dataset_formats import MatrixDataFrame

class Direction(Enum):
    # impute based on the median per user (column)
    USER = "user"
    # impute based on the median per sentence (row)
    SENTENCE = "sentence"
    # impute based on the median of the entire dataset
    GLOBAL = "global"

class MedianImputer(Imputer):
    '''Impute the data by using the median.
    If there's an even number of values, the median is the average of the two middle values.
    (This may lead to labels that are not part of the original dataset, which may need rounding later.)
    '''
    def __init__(self, name: str, direction: Direction, default_val=None, error_on_empty: bool=True):
        super().__init__(name)
        # USER = column
        # SENTENCE = row
        # GLOBAL = entire dataset
        self.direction: Direction = direction
        self.default_val = default_val
        # if error_on_empty is True, then throw an error when given an empty row or column to impute
        self.error_on_empty = error_on_empty

    def impute(self, df: MatrixDataFrame) -> Tuple[pd.DataFrame, JSONstr]:
        '''Impute the data with the median of the column, row, or entire dataset depending on the value of `self.direction`. If no self.default_val is given, then use the median of the entire dataset.
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # drop the sentence_id and sentence columns
            data_df = df.drop(columns=[SENTENCE_ID_COL, SENTENCE_COL])

            if self.default_val is None or self.direction == Direction.GLOBAL:
                # get the median of the entire dataset
                global_median: int = np.nanmedian(data_df.to_numpy(dtype=np.float64))
            
            if self.default_val is None:
                actual_default_val = global_median
            else:
                actual_default_val = self.default_val

            if self.direction == Direction.SENTENCE or self.direction == Direction.GLOBAL:
                # iterate through the df and replace all empty cells with the median for the row
                for i, row in data_df.iterrows():
                    if self.direction == Direction.GLOBAL:
                        median = global_median
                    else:
                        median = row.median()
                    if pd.isnull(median):
                        if self.error_on_empty:
                            raise ValueError(f"The {i}th (0-indexed) row of the given data was completely empty")
                        else:
                            median = actual_default_val
                    
                    data_df.loc[i] = row.fillna(median)
            elif self.direction == Direction.USER:
                # iterate through the df and replace all empty cells with the median for the column
                # skip the first two columns (sentence_id and sentence1)
                for i, col in data_df.items():
                    median = col.median()
                    if pd.isnull(median):
                        if self.error_on_empty:
                            raise ValueError(f"The column called '{i}' of the given data was completely empty")
                        else:
                            median = actual_default_val

                    data_df[i] = col.fillna(median)

            # add the sentence_id and sentence columns back
            data_df = data_df.copy()
            data_df[[SENTENCE_ID_COL, SENTENCE_COL]] = df[[SENTENCE_ID_COL, SENTENCE_COL]]
            assert set(data_df.columns) == set(df.columns), f"Columns of the original df and the imputed df are not the same. Original: {df.columns}, imputed: {data_df.columns}"
            data_df = data_df.reindex(columns=df.columns)

            empty_data = json.dumps({})
            return data_df, empty_data


if __name__ == "__main__":
    # create a fake df to test the imputer
    df = pd.DataFrame({"sentence_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "sentence1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "person_1": [2, 2, 4, 4, 5, None, None, None, None, None], "person_2": [8, None, None, 6, 6, 5, 5, 5, 5, 5], "person_3": [None, 4, 0, None, None, None, None, 1, None, None], "person_4": [None, None, None, None, None, None, None, None, None, None]})
    print("Original")
    print(df.to_string())
    # create an imputer
    imputer = MedianImputer("median_imputer", Direction.USER, error_on_empty=False)
    # impute the df
    imputed_df, _ = imputer.impute(df.copy())
    # print the imputed df
    print("User (column) median imputed")
    print(imputed_df.to_string())

    imputer = MedianImputer("median_imputer", Direction.SENTENCE)
    # impute the df
    imputed_df, _ = imputer.impute(df.copy())
    # print the imputed df
    print("Sentence (row) median imputed")
    print(imputed_df.to_string())