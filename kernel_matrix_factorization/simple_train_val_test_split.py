'''Given a df, return three dfs for train, validation and test sets'''
from typing import Tuple
import pandas as pd

def simple_train_val_test_split(df: pd.DataFrame, train_split=0.9, val_split=0.05) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''Split a df into train, validation and test sets. Test set is the remaining fraction after train and validation sets are taken.
    
    Note that the df does not need to be in any particular dataset format, such as matrix or factor.
    '''
    # don't modify the original df
    df = df.copy()
    
    # shuffle the df
    df = df.sample(frac=1).reset_index(drop=True)
    
    # get the number of rows in each set
    train_rows = int(train_split * len(df))
    val_rows = int(val_split * len(df))
    # unneeded - it will be the rest of the rows
    # test_rows = int(test_split * len(df))
    
    # split the df into train, validation and test sets
    train = df.iloc[:train_rows]
    val = df.iloc[train_rows:train_rows + val_rows]
    test = df.iloc[train_rows + val_rows:]
    
    return train, val, test
