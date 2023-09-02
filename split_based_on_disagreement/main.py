import numpy as np
import pandas as pd
import json
from typing import List
from sklearn.metrics import classification_report
import os
import warnings
import itertools

def make_path(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

# import from `compute_disagreement`
import sys
sys.path.append("../utilities/compute_disagreement")
from compute_disagreement import get_disagreement

# import from `kernel_matrix_factorization`
sys.path.append("../kernel_matrix_factorization")
from dataset_formats import NpyAnnotationsFormat

INDIVIDUAL_KEY = "individual"
CLASSIFICATION_REPORT_KEY = "cr"
WEIGHTED_AVG_KEY = "weighted avg"
F1_SCORE_KEY = "f1-score"
ACCURACY_KEY = "accuracy"
PREDICTIONS_KEY = "predictions"
TRUE_LABELS_KEY = "true_labels"
CHOSEN_EPOCH_KEY = "chosen_epoch"
HIGHEST_KEY = "choose_epoch_via_f1"
LOW_DISAGREEMENT_KEY = "low_disagreement"
MEDIUM_DISAGREEMENT_KEY = "medium_disagreement"
HIGH_DISAGREEMENT_KEY = "high_disagreement"
disagreement_keys = [LOW_DISAGREEMENT_KEY, MEDIUM_DISAGREEMENT_KEY, HIGH_DISAGREEMENT_KEY]
N_EXAMPLES_KEY = "n_examples"

def mad(data, axis=None):
    '''Mean absolute deviation from Stack Overflow'''
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def get_f1_score_from_epoch_dict(epoch_dict: dict):
    # we want the individual (not aggregate) f1 score
    return get_f1_score(epoch_dict[INDIVIDUAL_KEY][CLASSIFICATION_REPORT_KEY])

def get_f1_score(classification_report_dict: dict):
    return classification_report_dict[WEIGHTED_AVG_KEY][F1_SCORE_KEY]

def replace_values(arr: NpyAnnotationsFormat, replacement_list: List[int]):
    arr = arr.copy()
    assert np.sum(arr != -1) == len(replacement_list), f"Number of non-(-1)s in arr ({np.sum(arr != -1)}) does not match length of replacement_list ({len(replacement_list)})."
    mask = arr != -1
    arr[mask] = replacement_list[:np.count_nonzero(mask)]
    return arr

def main(args):
    # Load the .npy files and convert them to pandas DataFrames
    orig_data: NpyAnnotationsFormat = np.load(args.input_orig_data)
    orig_df = pd.DataFrame(orig_data)

    orig_disagreement = get_disagreement(orig_df)

    # Load the .json
    with open(args.trained_json, "r") as f:
        trained_json = json.load(f)
    
    # print(f"f1 scores: {[get_f1_score_from_epoch_dict(trained_json[epoch]) for epoch in trained_json]}")
    
    if args.highest:
        best_epoch = max(trained_json, key=lambda epoch: get_f1_score_from_epoch_dict(trained_json[epoch]))
        # print(f"The best epoch was {best_epoch}")
    else:
        best_epoch = max(trained_json, key=int)
        # print(f"The latest epoch was {best_epoch}")

    # print(f"Using epoch {best_epoch} with an individual weighted F1 score of {get_f1_score_from_epoch_dict(trained_json[best_epoch])}")

    imputed_predictions = trained_json[best_epoch][INDIVIDUAL_KEY][PREDICTIONS_KEY]
    imputed_true_labels = trained_json[best_epoch][INDIVIDUAL_KEY][TRUE_LABELS_KEY]

    # make sure that the predictions are equal to the number of non-(-1) values in the original data
    assert len(imputed_predictions) == np.sum(orig_data != -1), f"imputed_predictions has the wrong length: {len(imputed_predictions)} instead of {np.sum(orig_data != -1)}"

    # bin the proportions based on their value and print how many examples are in each bin
    # each value should have its own bin
    orig_disagreement_proportions = orig_disagreement[:, 2]

    # If you want to understand the distribution of disagreement proportions, uncomment the following lines
    proportions = np.unique(orig_disagreement_proportions)
    proportion_counts = np.array([np.sum(orig_disagreement_proportions == proportion) for proportion in proportions])
    # print(f"proportions: {proportions}")
    # print(f"proportion_counts: {proportion_counts}")
    # print(f"together: {sorted([(proportions[i], proportion_counts[i]) for i in range(len(proportions))], key=lambda x: x[0])}")

    low_disagreement_mask = None
    medium_disagreement_mask = None
    high_disagreement_mask = None
    if args.grouping_method == "std":
        mean_orig_disagreement = np.mean(orig_disagreement[:, 2])
        std_orig_disagreement = np.std(orig_disagreement[:, 2])

        assert isinstance(mean_orig_disagreement, float), f"mean_orig_disagreement is not a float: {mean_orig_disagreement}"
        assert isinstance(std_orig_disagreement, float), f"std_orig_disagreement is not a float: {std_orig_disagreement}"

        assert len(imputed_predictions) == len(imputed_true_labels), f"imputed_predictions and imputed_true_labels have different lengths: {len(imputed_predictions)} and {len(imputed_true_labels)}"

        # Split the data based on disagreement
        # Create a binary mask for the indices of the rows with low disagreement (disagreement < mean - std)
        low_disagreement_mask = orig_disagreement_proportions < mean_orig_disagreement - std_orig_disagreement
        high_disagreement_mask = orig_disagreement_proportions > mean_orig_disagreement + std_orig_disagreement
        medium_disagreement_mask = np.logical_not(np.logical_or(low_disagreement_mask, high_disagreement_mask))

    elif args.grouping_method == "cutoff":
        # low = less than 1/3 of people disagree
        low_disagreement_mask = orig_disagreement_proportions < (1/3)
        # medium = between 1/3 and 2/3 (inclusive) of people disagree
        medium_disagreement_mask = np.logical_and(orig_disagreement_proportions >= (1/3), orig_disagreement_proportions <= (2/3))
        # high = more than 2/3 of people disagree
        high_disagreement_mask = orig_disagreement_proportions > (2/3)

    elif args.grouping_method in ("mad", "var"):
        proportions = np.unique(orig_disagreement_proportions)
        proportion_counts = np.array([np.sum(orig_disagreement_proportions == proportion) for proportion in proportions])
        sorted_combined_proportions = sorted([(proportions[i], proportion_counts[i]) for i in range(len(proportions))], key=lambda x: x[0])

        # iterate through each possible low and high cutoff and find the one that minimizes the stat
        lowest_stat = float('inf')
        best_low_cutoff_index = None
        best_high_cutoff_index = None
        for low_cutoff_index, high_cutoff_index in itertools.combinations(range(len(sorted_combined_proportions)), 2):
            if low_cutoff_index + 2 > high_cutoff_index:
                # skip if the low cutoff is too close to the high cutoff
                continue

            low_cutoff_amount = sum(map(lambda x: x[1], sorted_combined_proportions[:low_cutoff_index + 1]))
            middle_cutoff_amount = sum(map(lambda x: x[1], sorted_combined_proportions[low_cutoff_index + 1: high_cutoff_index]))
            high_cutoff_amount = sum(map(lambda x: x[1], sorted_combined_proportions[high_cutoff_index:]))
            
            to_stat = np.array([low_cutoff_amount, middle_cutoff_amount, high_cutoff_amount])
            if args.grouping_method == "mad":
                stat_value = float(mad(to_stat))
            elif args.grouping_method == "var":
                stat_value = float(np.var(to_stat))
            else:
                raise ValueError("Invalid grouping method")

            # print(f"To stat: {to_stat}")
            # print(f"stat_value: {stat_value}")

            if stat_value < lowest_stat:
                lowest_stat = stat_value
                best_low_cutoff_index = low_cutoff_index
                best_high_cutoff_index = high_cutoff_index
                print("Found best!")

        assert best_low_cutoff_index is not None
        assert best_high_cutoff_index is not None

        low_cutoff_proportion = sorted_combined_proportions[best_low_cutoff_index][0]
        high_cutoff_proportion = sorted_combined_proportions[best_high_cutoff_index][0]

        # print(f"low_cutoff_proportion: {low_cutoff_proportion}")
        # print(f"high_cutoff_proportion: {high_cutoff_proportion}")
        # print(best_low_cutoff_index)
        # print(best_high_cutoff_index)

        low_disagreement_mask = orig_disagreement_proportions <= low_cutoff_proportion
        high_disagreement_mask = orig_disagreement_proportions >= high_cutoff_proportion
        medium_disagreement_mask = np.logical_not(np.logical_or(low_disagreement_mask, high_disagreement_mask))

    assert low_disagreement_mask is not None, "No low disagreement mask!"
    assert medium_disagreement_mask is not None, f"No medium disagreement mask!"
    assert high_disagreement_mask is not None, f"No high disagreement mask!"

    assert low_disagreement_mask.shape == (orig_disagreement.shape[0],), f"low_disagreement_mask has the wrong shape: {low_disagreement_mask.shape}"
    assert medium_disagreement_mask.shape == (orig_disagreement.shape[0],), f"medium_disagreement_mask has the wrong shape: {medium_disagreement_mask.shape}"
    assert high_disagreement_mask.shape == (orig_disagreement.shape[0],), f"high_disagreement_mask has the wrong shape: {high_disagreement_mask.shape}"


    if np.sum(low_disagreement_mask) == 0:
        print("No low disagreement rows!")
    if np.sum(medium_disagreement_mask) == 0:
        print("No medium disagreement rows!")
    if np.sum(high_disagreement_mask) == 0:
        print("No high disagreement rows!")

    # print(f"Number of rows with low disagreement: {np.sum(low_disagreement_mask)}")
    # print(f"Number of rows with medium disagreement: {np.sum(medium_disagreement_mask)}")
    # print(f"Number of rows with high disagreement: {np.sum(high_disagreement_mask)}")

    # reshape the prediction arrays into the format of the original data, with the predictions put in where the original data (the non-(-1) values) are
    imputed_predictions = replace_values(orig_data, imputed_predictions)
    imputed_true_labels = replace_values(orig_data, imputed_true_labels)

    # Get the predictions and true labels for the low disagreement data
    low_disagreement_predictions = imputed_predictions[low_disagreement_mask]
    low_disagreement_true_labels = imputed_true_labels[low_disagreement_mask]

    assert low_disagreement_predictions.shape == low_disagreement_true_labels.shape, f"low_disagreement_predictions and low_disagreement_true_labels have different shapes: {low_disagreement_predictions.shape} and {low_disagreement_true_labels.shape}"
    assert low_disagreement_predictions.shape == (np.sum(low_disagreement_mask), orig_data.shape[1]), f"low_disagreement_predictions has the wrong shape: {low_disagreement_predictions.shape} instead of {(np.sum(low_disagreement_mask), orig_data.shape[1])}"

    # remove the -1s and flatten back into list
    low_disagreement_predictions = low_disagreement_predictions[low_disagreement_predictions != -1].flatten().tolist()
    low_disagreement_true_labels = low_disagreement_true_labels[low_disagreement_true_labels != -1].flatten().tolist()

    assert len(low_disagreement_predictions) == len(low_disagreement_true_labels), f"low_disagreement_predictions and low_disagreement_true_labels have different lengths: {len(low_disagreement_predictions)} and {len(low_disagreement_true_labels)}"

    # Get the predictions and true labels for the medium disagreement data
    medium_disagreement_predictions = imputed_predictions[medium_disagreement_mask]
    medium_disagreement_true_labels = imputed_true_labels[medium_disagreement_mask]
    medium_disagreement_predictions = medium_disagreement_predictions[medium_disagreement_predictions != -1].flatten().tolist()
    medium_disagreement_true_labels = medium_disagreement_true_labels[medium_disagreement_true_labels != -1].flatten().tolist()

    # Get the predictions and true labels for the high disagreement data
    high_disagreement_predictions = imputed_predictions[high_disagreement_mask]
    high_disagreement_true_labels = imputed_true_labels[high_disagreement_mask]
    high_disagreement_predictions = high_disagreement_predictions[high_disagreement_predictions != -1].flatten().tolist()
    high_disagreement_true_labels = high_disagreement_true_labels[high_disagreement_true_labels != -1].flatten().tolist()

    assert len(low_disagreement_predictions) + len(medium_disagreement_predictions) + len(high_disagreement_predictions) == np.sum(orig_data != -1), f"low_disagreement_predictions, medium_disagreement_predictions, and high_disagreement_predictions don't add up to the number of non-(-1) values in the original data: {len(low_disagreement_predictions)}, {len(medium_disagreement_predictions)}, {len(high_disagreement_predictions)} which total to {len(low_disagreement_predictions) + len(medium_disagreement_predictions) + len(high_disagreement_predictions)} instead of {np.sum(orig_data != -1)}"

    return_data = {
        CHOSEN_EPOCH_KEY: best_epoch,
        HIGHEST_KEY: args.highest,
        LOW_DISAGREEMENT_KEY: {
            PREDICTIONS_KEY: low_disagreement_predictions,
            TRUE_LABELS_KEY: low_disagreement_true_labels,
            N_EXAMPLES_KEY: int(np.sum(low_disagreement_mask))
        },
        MEDIUM_DISAGREEMENT_KEY: {
            PREDICTIONS_KEY: medium_disagreement_predictions,
            TRUE_LABELS_KEY: medium_disagreement_true_labels,
            N_EXAMPLES_KEY: int(np.sum(medium_disagreement_mask))
        },
        HIGH_DISAGREEMENT_KEY: {
            PREDICTIONS_KEY: high_disagreement_predictions,
            TRUE_LABELS_KEY: high_disagreement_true_labels,
            N_EXAMPLES_KEY: int(np.sum(high_disagreement_mask))
        }
    }

    # Compute the classification report for each of the three subsets
    for disagreement_key in disagreement_keys:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return_data[disagreement_key][CLASSIFICATION_REPORT_KEY] = classification_report(return_data[disagreement_key][TRUE_LABELS_KEY], return_data[disagreement_key][PREDICTIONS_KEY], output_dict=True, zero_division=0)

    # save the data
    make_path(args.output_json)
    with open(args.output_json, "w") as f:
        json.dump(return_data, f, indent=4)

    return return_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_orig_data", type=str, help="Path to the original .npy file")
    parser.add_argument("--trained_json", type=str, help="Path to the .json file from training")
    parser.add_argument("--output_json", type=str, help="Path to the output .json file")
    parser.add_argument("--grouping_method", type=str, choices=["std", "cutoff", "var", "mad"], help="How to group the data into low, medium, and high disagreement. std splits based on standard deviation, cutoff splits based on even cutoff values, var splits the data evenly across the three categories (as much as possible, based on variance), and mad also splits the data as evenly as possible, but based on Mean Absolute Deviation")
    parser.add_argument("--highest", action="store_true", help="Whether to use the validation with the highest weighted F1 score (default: use data from last epoch)") # We set this to true when reporting results

    args = parser.parse_args()

    # if they don't pass anything, run the full testing
    quick_test = False
    if args.input_orig_data == args.trained_json == args.output_json == None:
        quick_test = True

    if quick_test:
        quick_test_data = {}
        from tqdm import tqdm
        dataset_names = ["SChem", "ghc", "politeness", "Sentiment"]
        # dataset_names = ["ghc"] # quick test
        folds = [0, 1, 2, 3, 4]
        original_or_imputed_list = ["original", "imputed"]
        for dataset_name in tqdm(dataset_names, desc="Dataset"):
            quick_test_data[dataset_name] = {"overall": {}}
            for fold in tqdm(folds, desc="Fold", leave=False):
                quick_test_data[dataset_name][fold] = {}
                for original_or_imputed in tqdm(original_or_imputed_list, desc="Original or Imputed", leave=False):
                    # for better printing inside tqdm
                    print = lambda thing: tqdm.write(str(thing))
                    args.input_orig_data = f"../datasets/cleaned_folds/{dataset_name}/{dataset_name}_val_{fold}_annotations.npy"
                    args.trained_json = f"../datasets/post_arxiv_run1_data/logs/{dataset_name}_{original_or_imputed}_{fold}_full_run1.json"
                    args.output_json = f"outputs/{original_or_imputed}/{dataset_name}_F{fold}_disagreement_analysis.json"
                    # std, cutoff, split
                    args.grouping_method = "var"
                    args.highest = True

                    individual_data = main(args)
                    quick_test_data[dataset_name][fold][original_or_imputed] = individual_data
        # gather all of the results across the folds for n_examples, weighted_f1, and accuracy
        for disagreement_key in disagreement_keys:
            for dataset_name in tqdm(dataset_names, desc="Dataset"):
                quick_test_data[dataset_name]["overall"][disagreement_key] = {}
                overall_stats = {}
                for original_or_imputed in tqdm(original_or_imputed_list, desc="Original or Imputed", leave=False):
                    overall_stats[original_or_imputed] = {
                        "dataset_n_examples": [],
                        "dataset_weighted_f1": [],
                        "dataset_accuracy": []
                    }
                    for fold in tqdm(folds, desc="Fold", leave=False):
                        overall_stats[original_or_imputed]["dataset_n_examples"].append(quick_test_data[dataset_name][fold][original_or_imputed][disagreement_key][N_EXAMPLES_KEY])
                        overall_stats[original_or_imputed]["dataset_weighted_f1"].append(get_f1_score(quick_test_data[dataset_name][fold][original_or_imputed][disagreement_key][CLASSIFICATION_REPORT_KEY]))
                        overall_stats[original_or_imputed]["dataset_accuracy"].append(quick_test_data[dataset_name][fold][original_or_imputed][disagreement_key][CLASSIFICATION_REPORT_KEY][ACCURACY_KEY])

                    # Compute std of the weighted f1 scores and accuracy
                    dataset_weighted_f1_std = float(np.std(overall_stats[original_or_imputed]["dataset_weighted_f1"]))
                    dataset_accuracy_std = float(np.std(overall_stats[original_or_imputed]["dataset_accuracy"]))

                    # Compute the mean of the weighted f1 scores and accuracy
                    dataset_weighted_f1 = float(np.mean(overall_stats[original_or_imputed]["dataset_weighted_f1"]))
                    dataset_accuracy = float(np.mean(overall_stats[original_or_imputed]["dataset_accuracy"]))

                    dataset_n_examples = int(np.sum(overall_stats[original_or_imputed]["dataset_n_examples"]))

                    quick_test_data[dataset_name]["overall"][disagreement_key][original_or_imputed] = {
                        N_EXAMPLES_KEY: dataset_n_examples,
                        "weighted_f1_mean": dataset_weighted_f1,
                        "weighted_f1_std": dataset_weighted_f1_std,
                        "accuracy_mean": dataset_accuracy,
                        "accuracy_std": dataset_accuracy_std
                    }

        # save the data
        all_outputs_path = f"outputs/quick_test_disagreement_analysis.json"
        make_path(all_outputs_path)
        with open(all_outputs_path, "w") as f:
            json.dump(quick_test_data, f, indent=4)

        # save just the overall results
        overall_results = {}
        for dataset_name in dataset_names:
            overall_results[dataset_name] = quick_test_data[dataset_name]["overall"]

        overall_outputs_path = f"outputs/quick_test_disagreement_analysis_overall.json"
        make_path(overall_outputs_path)
        with open(overall_outputs_path, "w") as f:
            json.dump(overall_results, f, indent=4)
    else:
        main(args)