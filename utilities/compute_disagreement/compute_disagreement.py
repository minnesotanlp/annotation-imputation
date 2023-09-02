import argparse
import numpy as np
import pandas as pd
from scipy.stats import mode

def get_disagreement(df: pd.DataFrame):
    '''Given a df of ratings (each row is an example, each column is an annotator, each value is a rating), return a numpy array where the first column is the number of people who gave a rating that disagreed with the majority rating, the second column is the number of people who gave a rating for that example, and the third column is the proportion of people who gave a rating that disagreed with the majority rating.'''
    df = df.copy()
    # Replace -1 (missing ratings) with NaN
    df.replace(-1, np.nan, inplace=True)

    # Calculate the mode and its count for each row using scipy.stats.mode()
    mode_result = mode(df, axis=1, nan_policy='omit', keepdims=True)
    # print(f"Mode result: {mode_result}")

    # mode_values = mode_result.mode
    mode_counts = mode_result.count
    # print(f"Mode counts: {mode_counts}")

    # Calculate the number of non-missing ratings for each row
    count = df.count(axis=1)
    # print(f"Count: {count}")

    # Calculate the number of people who disagree with the majority rating
    disagree = count - mode_counts.flatten()
    # print(f"Disagree: {disagree}")

    # Calculate the proportion of people who gave disagreeing ratings divided by how many people gave ratings for that example
    proportion = disagree / count
    # print(f"Proportion: {proportion}")

    # Create a new numpy array with the shape (n, 3) containing the calculated values
    result = np.column_stack((disagree, count, proportion))
    return result

def main(input_file, output_file):
    # Load the .npy file and convert it to a pandas DataFrame
    data = np.load(input_file)
    df = pd.DataFrame(data)

    result = get_disagreement(df)

    # Save the new numpy array as a .npy file
    np.save(output_file, result)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computes the disagreement within a .npy file with annotations. Outputs a .npy file where the first column is the number of people who gave a rating that disagreed with the majority rating, the second column is the number of people who gave a rating for that example, and the third column is the proportion of people who gave a rating that disagreed with the majority rating.")
    parser.add_argument("input_file", type=str, help="Path to the .npy file")
    parser.add_argument("output_file", type=str, help="Path to the output .npy file")

    args = parser.parse_args()

    main(args.input_file, args.output_file)