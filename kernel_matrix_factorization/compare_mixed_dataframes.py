import numpy as np
def compare_mixed_dataframes(df1, df2) -> bool:
    # Written by GPT-4
    # Get the column names of numerical columns
    num_cols = df1.select_dtypes(include=[np.number]).columns
    
    # Convert numerical columns to float and replace None with NaN
    df1_num = df1[num_cols].astype(float).fillna(np.nan)
    df2_num = df2[num_cols].astype(float).fillna(np.nan)

    # Compare numerical columns with a tolerance value using numpy.allclose()
    num_comparison = np.allclose(df1_num, df2_num, rtol=1e-05, atol=1e-08, equal_nan=True)

    # Compare sentence columns using pandas.DataFrame.equals()
    string_cols = df1.select_dtypes(include=['object']).columns
    str_comparison = df1[string_cols].equals(df2[string_cols])

    # Combine the results of numerical and sentence columns comparisons
    return num_comparison and str_comparison