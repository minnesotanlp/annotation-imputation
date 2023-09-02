import pandas as pd
from typing import Tuple
from assert_valid_factor_data_frame import assert_valid_factor_data_frame
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report

# imports from `kernel_matrix_factorization`
import sys
sys.path.append("../kernel_matrix_factorization")
from dataset_columns import LABEL_COL, SENTENCE_ID_COL, USER_ID_COL
from dataset_formats import FactorDataFrame

def grab_examples(goal_df: FactorDataFrame, source_df: FactorDataFrame, all_goal=False) -> Tuple[FactorDataFrame, FactorDataFrame]:
    '''Grabs all of the examples that are in both the goal dataframe and the source dataframe'''
    # assert_valid_factor_data_frame(goal_df)
    # assert_valid_factor_data_frame(source_df)

    suffixes = ('_goal', '_source')
    suffixed_dfs = [df.add_suffix(suffix) for df, suffix in zip([goal_df, source_df], suffixes)]

    sentence_id_col = [SENTENCE_ID_COL + suffix for suffix in suffixes]
    user_id_col = [USER_ID_COL + suffix for suffix in suffixes]
    
    # Merge the two dataframes on the specified columns
    merged_df = suffixed_dfs[0].merge(suffixed_dfs[1], left_on=[sentence_id_col[0], user_id_col[0]], right_on=[sentence_id_col[1], user_id_col[1]])

    # Split the merged dataframe back into the two match dataframes
    goal_match_df = merged_df[[col for col in merged_df.columns if col.endswith(suffixes[0])]].rename(columns=lambda x: x[:-len(suffixes[0])])
    source_match_df = merged_df[[col for col in merged_df.columns if col.endswith(suffixes[1])]].rename(columns=lambda x: x[:-len(suffixes[1])])

    return goal_match_df, source_match_df

def get_predicted_and_test_labels(source_df: FactorDataFrame, test_df: FactorDataFrame, round: bool=False) -> Tuple[pd.Series, pd.Series]:
    '''Gets the predicted and test labels of a source dataframe which contains examples from a test dataframe'''
    test_examples: FactorDataFrame
    predicted_examples: FactorDataFrame
    test_examples, predicted_examples = grab_examples(goal_df=test_df, source_df=source_df)
    test_labels = test_examples[LABEL_COL]
    predicted_labels = predicted_examples[LABEL_COL]

    if round:
        predicted_labels = predicted_labels.round()
        previous_test_labels = test_labels
        test_labels = test_labels.round()
        assert (previous_test_labels == test_labels).all(), "Expected test labels to be the same after rounding"

    return predicted_labels, test_labels

def get_rmse_score(source_df: FactorDataFrame, test_df: FactorDataFrame):
    '''Gets the RMSE score of a source dataframe which contains examples from a test dataframe'''
    predicted_labels, test_labels = get_predicted_and_test_labels(source_df=source_df, test_df=test_df, round=False)
    # print("test labels")
    # print(test_labels)
    # print("predicted labels")
    # print(predicted_labels)
    rmse_score = mean_squared_error(test_labels, predicted_labels, squared=False)
    # print("rmse_score:")
    # print(rmse_score)
    return rmse_score

def get_classification_report(source_df: FactorDataFrame, test_df: FactorDataFrame):
    predicted_labels, test_labels = get_predicted_and_test_labels(source_df=source_df, test_df=test_df, round=True)
    return classification_report(test_labels, predicted_labels, zero_division=0, output_dict=True)