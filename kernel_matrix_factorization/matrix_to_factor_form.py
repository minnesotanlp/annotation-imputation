'''
Given a df in matrix form, return a df in factor form.
See dataset_types.md for info on the different forms.
'''

from dataset_columns import SENTENCE_ID_COL, SENTENCE_COL, USER_ID_COL, LABEL_COL
from dataset_formats import MatrixDataFrame, FactorDataFrame, MissingDataFrame
from typing import Union, Tuple

def matrix_to_factor_form(df: MatrixDataFrame, get_missing_ratings=False) -> Union[FactorDataFrame, Tuple[FactorDataFrame, MissingDataFrame]]:
    '''
    Given a df in matrix form, return a df in factor form.
    Matrix form has one row per item.
    The first two columns are the item_id and item.
    The remaining columns are the user_ids.
    The values in the cells are the ratings.

    Factor form has one row per rating.

    If get_missing_ratings is True, then also return a df containing all of the rows from the input df that have missing ratings.

    Does not modify the original given dataframe
    '''
    # copy the df so we don't modify the original
    df = df.copy()

    # the pandas melt function is useful for this
    df = df.melt(id_vars=[SENTENCE_ID_COL, SENTENCE_COL], var_name=USER_ID_COL, value_name=LABEL_COL)

    if get_missing_ratings:
        # instead of dropping the rows, keep them in a separate df
        empty_ratings_df = df[df[LABEL_COL].isna()]
        empty_ratings_df = empty_ratings_df[[SENTENCE_ID_COL, SENTENCE_COL, USER_ID_COL]]
        empty_ratings_df = empty_ratings_df.reset_index(drop=True)
    
    # drop rows where the rating is missing
    df = df.dropna(subset=[LABEL_COL])

    # reset the index
    df = df.reset_index(drop=True)

    if get_missing_ratings:
        return df, empty_ratings_df
    else:
        return df

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('../datasets/matrix/SChem_matrix.csv')
    print(df.head())

    df, empty_df = matrix_to_factor_form(df, get_missing_ratings=True)
    print(df.head())
    print(empty_df.head())

    df.to_csv('../testing/SChem_factor_matrix_to_factor.csv', index=False)
    empty_df.to_csv('../testing/SChem_factor_matrix_to_factor_empty.csv', index=False)