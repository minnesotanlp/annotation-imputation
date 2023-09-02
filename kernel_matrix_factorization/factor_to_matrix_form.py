'''
Given a df in factor form, return a df in matrix form.
See dataset_types.md for info on the different forms.
'''

import pandas as pd
from dataset_columns import USER_ID_COL, SENTENCE_ID_COL, LABEL_COL, SENTENCE_COL
from dataset_formats import MatrixDataFrame, FactorDataFrame
import warnings

def factor_to_matrix_form(df: FactorDataFrame) -> MatrixDataFrame:
    '''
    Take a df in factor form (with only the columns for the user_id, sentence_id, sentence, and label) and turn it into matrix form.
    Does not modify the input df.
    Note that if the factor form df is missing an annotator or text, that column or row will be dropped in the matrix form df.
    This means that converting from matrix to factor and back to matrix may result in a different df.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        df_with_item = df.copy() # save the item col for later
        df = df.drop(columns=[SENTENCE_COL])

        # make sure the df is in the right format
        assert set(df.columns) == {USER_ID_COL, SENTENCE_ID_COL, LABEL_COL}, df.columns

        # pivot the df
        df = df.pivot(index=SENTENCE_ID_COL, columns=USER_ID_COL, values=LABEL_COL)

        # assert that there's no duplicate item_id
        assert len(df.index) == len(set(df.index)), "There are duplicate item_id values."

        # get the corresponding item for each item_id by getting the first row of the old df with that item_id
        item_df = df_with_item[[SENTENCE_ID_COL, SENTENCE_COL]].drop_duplicates(subset=[SENTENCE_ID_COL])
        item_df = item_df.set_index(SENTENCE_ID_COL)

        # put the items in the first non-user_id column
        df.insert(0, SENTENCE_COL, item_df[SENTENCE_COL])

        # reset the index
        df = df.reset_index()

        return df
    
def factor_to_matrix_form_with_reference(df: FactorDataFrame, reference_df: MatrixDataFrame, all_none=True):
    '''Convert the given df to matrix form, but use the reference_df to insert columns and rows that are missing from the converted df.
    The exact values in the reference_df are used, meaning that Nones and NaN are treated the same. If all_none is True, then we convert all NaNs to Nones.
    '''
    # convert the df to matrix form
    df = factor_to_matrix_form(df)

    # get the columns that are missing from the converted df
    missing_columns = set(reference_df.columns) - set(df.columns)

    assert len(missing_columns) == reference_df.shape[1] - df.shape[1], f"The number of missing columns is not correct. Missing columns len: {len(missing_columns)}. Reference df shape: {reference_df.shape}. Converted df shape: {df.shape}."

    # add the missing column names (not the data) to the converted df
    # for column in missing_columns:
    #     df[column] = None
    df = df.reindex(columns=list(df.columns) + list(missing_columns))

    # assert that the number of columns are the same
    assert len(df.columns) == len(reference_df.columns), f"The number of columns are not the same. Converted df columns: {df.columns}. Reference df columns: {reference_df.columns}."

    # sort the columns
    df = df[reference_df.columns]

    # get the rows that are missing from the converted df
    missing_rows = set(reference_df[SENTENCE_ID_COL]) - set(df[SENTENCE_ID_COL])

    assert len(missing_rows) == reference_df.shape[0] - df.shape[0], f"The number of missing rows is not correct. Missing rows len: {len(missing_rows)}. Reference df shape: {reference_df.shape}. Converted df shape: {df.shape}."

    # add the missing rows' SENTENCE_ID_COL and SENTENCE_COL to the converted df
    missing_rows_df = reference_df[reference_df[SENTENCE_ID_COL].isin(missing_rows)][[SENTENCE_ID_COL, SENTENCE_COL]]
    assert len(missing_rows_df) == len(missing_rows), f"The lengths are not the same. Missing rows df length: {len(missing_rows_df)}. Missing rows length: {len(missing_rows)}."
    df = pd.concat([df, missing_rows_df], ignore_index=True)

    # assert that the dimensions are the same
    assert df.shape == reference_df.shape, f"The dimensions are not the same. Converted df shape: {df.shape}. Reference df shape: {reference_df.shape}."

    # get the rows to match the order of the reference df
    df = df.sort_values(by=[SENTENCE_ID_COL])

    # reset the index
    df = df.reset_index(drop=True)

    # make the datatypes the same
    df = df.astype(reference_df.dtypes.to_dict())

    if all_none:
        # convert all NaNs to Nones
        df = df.where(pd.notnull(df), None)

    # assert that the dimensions are the same
    assert df.shape == reference_df.shape, f"The dimensions are not the same. Converted df shape: {df.shape}. Reference df shape: {reference_df.shape}."

    return df

if __name__ == "__main__":
    import pandas as pd
    from dataset_columns import USER_ID_COL, SENTENCE_ID_COL, LABEL_COL, SENTENCE_COL
    from dataset_formats import MatrixDataFrame, FactorDataFrame
    from matrix_to_factor_form import matrix_to_factor_form

    # Create a small dataset in matrix form
    data = {
        SENTENCE_ID_COL: [1, 2, 3, 4],
        SENTENCE_COL: ["This is a sentence.", "Another sentence here.", "Yet another sentence.", "Again a sentence."],
        "user_1": [1, None, 2, 3],
        "user_2": [None, None, 1, 2],
        "user_3": [None, None, None, None],
        "user_4": [4, None, 6, 3]
    }

    matrix_df = pd.DataFrame(data, dtype="object")
    print("Matrix form:")
    print(matrix_df)

    # Convert the matrix form to factor form
    factor_df = matrix_to_factor_form(matrix_df)
    # print("Factor form:")
    # print(factor_df)

    # Convert the factor form back to matrix form using the factor_to_matrix_form_with_reference function
    converted_matrix_df = factor_to_matrix_form_with_reference(factor_df, matrix_df)
    print("Converted matrix form:")
    print(converted_matrix_df)

    # converted_matrix_df2 = factor_to_matrix_form(factor_df)
    # print("Converted matrix form 2:")
    # print(converted_matrix_df2)

    # Check if the resulting matrix form is the same as the original matrix form
    assert matrix_df.equals(converted_matrix_df), "The resulting matrix form is not the same as the original matrix form."

    print("Test passed!")