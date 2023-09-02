'''
Splitter that grabs one rating from each example that has more than one rating.
Then, imputers can be trained on the remaining ratings.
'''

from assert_valid_matrix_data_frame import assert_valid_matrix_data_frame
from dataset_splitter import DatasetSplitter
from typing import Sequence
from tqdm import tqdm
import numpy as np
from math import isclose

# imports from Kernel Matrix Factorization
import sys
sys.path.append("../kernel_matrix_factorization")
from dataset_formats import MatrixDataFrame
from dataset_columns import SENTENCE_ID_COL, SENTENCE_COL
from one_anno_per_cross_split import factor_to_factor_one_anno_per_cross_split
from dataset_format_conversions import factor_to_matrix_form, matrix_to_factor_form, factor_to_matrix_form_with_reference


class TwoOrMoreRowImputerTestSplitter(DatasetSplitter):
    def __init__(self, name: str = "TwoOrMoreRowImputerTestSplitter"):
        super().__init__(name)

    def split(self, df: MatrixDataFrame) -> Sequence[MatrixDataFrame]:
        assert_valid_matrix_data_frame(df)
        orig_df = df.copy()
        train_df = df.copy()
        test_df = df.copy()

        # iterate through the rows of the df
        for row_index, row in tqdm(orig_df.iterrows(), total=len(df)):
            # if the row has more than one rating
            if len(row.dropna()) > 2:
                # pick one of the ratings at random (excluding empty ratings)
                columns_to_choose_from = row.dropna().index.tolist()
                columns_to_choose_from.remove(SENTENCE_ID_COL)
                columns_to_choose_from.remove(SENTENCE_COL)
                chosen_column = np.random.choice(columns_to_choose_from)
                # remove the chosen rating from the train df
                train_df.at[row_index, chosen_column] = None
                # remove all other ratings from the test df
                for column in columns_to_choose_from:
                    if column != chosen_column:
                        test_df.at[row_index, column] = None
        assert_valid_matrix_data_frame(train_df)
        assert_valid_matrix_data_frame(test_df)
        print("Split complete!")
        # print(f"Train shape: {train_df.shape}")
        # print(f"Test shape: {test_df.shape}")
        return train_df, test_df
    
class OnePerCrossImputerTestSplitter(DatasetSplitter):
    def __init__(self, name: str = "OnePerCrossImputerTestSplitter", val_split: float=0.5, seed: int=0):
        super().__init__(name)
        self.val_split = val_split
        self.seed = seed

    def split(self, df: MatrixDataFrame) -> Sequence[MatrixDataFrame]:
        assert_valid_matrix_data_frame(df)
        
        # convert to factor format
        factor_df = matrix_to_factor_form(df)

        # split into train and validation (test is not used)
        train_split = 1 - self.val_split
        train_factor_df, val_factor_df, _test_factor_df = factor_to_factor_one_anno_per_cross_split(factor_df, train_split=train_split, val_split=self.val_split, seed=self.seed)
        
        # it's okay to be off by one because of rounding
        assert isclose(len(train_factor_df) + len(val_factor_df), len(factor_df), abs_tol=1), f"{len(train_factor_df)} + {len(val_factor_df)} != {len(factor_df)}"
        assert isclose(len(_test_factor_df), 0, abs_tol=1), f"Test df should be empty, but has {len(_test_factor_df)} rows"

        # convert back to matrix format, with the original df as a reference
        train_df = factor_to_matrix_form_with_reference(train_factor_df, df)
        val_df = factor_to_matrix_form_with_reference(val_factor_df, df)
        assert_valid_matrix_data_frame(train_df)
        assert_valid_matrix_data_frame(val_df)
        assert train_df.shape == df.shape, f"{train_df.shape} != {df.shape}"
        print("Split complete!")

        return train_df, val_df
        

if __name__ == "__main__":
    import argparse
    from dataclasses import replace
    from dataset_splitter import split_npy
    
    # imports from Kernel Matrix Factorization
    import sys
    sys.path.append("../kernel_matrix_factorization")
    from save_df import save_df

    parser = argparse.ArgumentParser()
    # for both splitters
    parser.add_argument("--annotations", type=str, required=True, help="Path to the npy file containing the annotations")
    parser.add_argument("--texts", type=str, required=True, help="Path to the npy file containing the texts")
    parser.add_argument("--train_output", type=str, required=True, help="Path to the output file for the train data")
    parser.add_argument("--test_output", type=str, required=True, help="Path to the output file for the test data")

    # for OnePerCrossImputerTestSplitter
    parser.add_argument("--test_split", type=float, default=0.05, help="Fraction of the data to use for testing (1=100%)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed to use for splitting")
    
    args = parser.parse_args()
    annotations = np.load(args.annotations, allow_pickle=True)
    texts = np.load(args.texts, allow_pickle=True)

    # splitter = TwoOrMoreRowImputerTestSplitter()
    splitter = OnePerCrossImputerTestSplitter(val_split=args.test_split, seed=args.seed)

    print("May take a bit (<1 minute) to convert the npy files to dataframes. Please be patient.")
    train_df, test_df = split_npy(splitter, annotations, texts)
    assert_valid_matrix_data_frame(train_df)
    assert_valid_matrix_data_frame(test_df)
    save_df(train_df, args.train_output)
    save_df(test_df, args.test_output)
    print(f"Saved train data to {args.train_output}")
    print(f"Saved test data to {args.test_output}")