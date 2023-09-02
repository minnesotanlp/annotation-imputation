'''
Recreate the variance graph from Ruyuan
Each point represents text.
'''
import matplotlib.pyplot as plt
import argparse
import numpy as np
import json
import os

def make_path(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

def get_variance(annotations):
    annotations = annotations.copy()
    annotations = annotations.astype(float)
    # convert -1 to np.nan
    annotations[annotations == -1] = np.nan
    # calculate variance
    variance = np.nanvar(annotations, axis=1)
    return variance

def get_majority_vote(annotations):
    annotations = annotations.copy()
    # add one to all elements so -1 becomes 0
    annotations += 1

    # get the bincount along axis 1 for the unique values
    # equivalent to bincounts = np.array([np.bincount(row, minlength=np.max(annotations)+1) for row in annotations])
    bincounts = np.apply_along_axis(np.bincount, axis=1, arr=annotations, minlength=np.max(annotations)+1)

    # set the count for 0 to 0
    bincounts[:, 0] = 0

    # get the majority vote, and subtract 1 so 0 becomes -1 again
    maj_vote = np.argmax(bincounts, axis=1) - 1

    return maj_vote

def get_disagreement_rate(annotations, majority_voted):
    # print("Annotations:")
    # print(annotations)
    # print("Majority voted:")
    # print(majority_voted)
    # Create a mask to ignore -1 annotations
    mask = annotations != -1

    # Compute the number of annotations that disagree with the majority vote for each row
    disagree = np.sum((annotations != majority_voted[:, np.newaxis]) & mask, axis=1)

    # Compute the total number of annotations (ignoring -1) for each row
    total_annotations = np.sum(mask, axis=1)

    # Compute the ratio for each row
    ratio = disagree / total_annotations

    # print("Disagreeing annotations:", disagree)
    # print("Total annotations:", total_annotations)
    # print("Ratio:", ratio)
    return ratio

def main(args):
    orig_annotations = np.load(args.original_annotations)
    imputed_annotations = np.load(args.imputed_annotations)
    # convert both to int
    orig_annotations = orig_annotations.astype(int)
    imputed_annotations = imputed_annotations.astype(int)
    assert orig_annotations.shape == imputed_annotations.shape

    # Calculate variance for each row, ignoring -1
    orig_variance = get_variance(orig_annotations)
    imputed_variance = get_variance(imputed_annotations)

    # # print the row index of the highest variance example for orig
    # print("Max variance for orig:")
    # print(np.argmax(orig_variance), orig_variance[np.argmax(orig_variance)])
    # print(orig_annotations[np.argmax(orig_variance)])

    # Calculate normalized disagreement rate

    # First, calculate the majaority vote
    # optimized without for loop
    orig_maj_vote = get_majority_vote(orig_annotations)
    imputed_maj_vote = get_majority_vote(imputed_annotations)


    # Calculate the disagreement rate
    # this is the number of annotations that disagree with the majority vote divided by the number of annotations (not -1)
    orig_disagreement_rate = get_disagreement_rate(orig_annotations, orig_maj_vote)
    imputed_disagreement_rate = get_disagreement_rate(imputed_annotations, imputed_maj_vote)

    # Plot
    plt.scatter(imputed_disagreement_rate, imputed_variance, label='Imputed')
    plt.scatter(orig_disagreement_rate, orig_variance, label='Original')
    label_font_size = 16
    title_font_size = 18
    plt.xlabel('Disagreement Rate', fontsize=label_font_size)
    plt.ylabel('Variance', fontsize=label_font_size)
    plt.title(f'Disagreement Rate vs Variance for {args.dataset}', fontsize=title_font_size)
    plt.legend()
    make_path(args.graph_output)
    plt.savefig(args.graph_output)
    # plt.show()
    plt.clf()

    # compute average variance and disagreement rate
    original_average_variance = np.mean(orig_variance)
    imputed_average_variance = np.mean(imputed_variance)

    original_average_disagreement_rate = np.mean(orig_disagreement_rate)
    imputed_average_disagreement_rate = np.mean(imputed_disagreement_rate)

    data = {
        'Average Variance': {
            'Original': original_average_variance,
            'Imputed': imputed_average_variance,
            'Imputed - Original': imputed_average_variance - original_average_variance,
            'Imputed - Original (Rounded)': round(imputed_average_variance - original_average_variance, 3)
        },
        'Average Disagreement Rate': {
            'Original': original_average_disagreement_rate,
            'Imputed': imputed_average_disagreement_rate,
            'Imputed - Original': imputed_average_disagreement_rate - original_average_disagreement_rate,
            'Imputed - Original (Rounded)': round(imputed_average_disagreement_rate - original_average_disagreement_rate, 3)
        }
    }

    make_path(args.json_output)
    with open(args.json_output, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--original_annotations', type=str, help='Path to original annotations')
    parser.add_argument('--imputed_annotations', type=str, help='Path to imputed annotations')
    parser.add_argument('--graph_output', type=str, help='Path to output (PNG) graph')
    parser.add_argument('--json_output', type=str, help='Path to output json data')

    args = parser.parse_args()

    if args.dataset == args.original_annotations == args.imputed_annotations == args.graph_output == args.json_output and args.dataset is None:
        print("No arguments detected! Running all datasets.")
        # run everything
        # {display_name: actual_dataset_name}
        datasets = {"GHC": "ghc", "SBIC": "Fixed2xSBIC", "Politeness": "politeness", "SChem": "SChem", "SChem5Labels": "SChem5Labels", "Sentiment": "Sentiment"}

        for display_name, actual_dataset_name in datasets.items():
            args.dataset = display_name
            args.original_annotations = f'../datasets/final_imputation_results/original/{actual_dataset_name}_annotations.npy'
            args.imputed_annotations = f'../datasets/final_imputation_results/ncf_imputation_results/{actual_dataset_name}_-1_ncf_imputation_-1.npy'
            args.graph_output = f'outputs/graphs/{actual_dataset_name}_graph.png'
            args.json_output = f'outputs/json/{actual_dataset_name}_data.json'

            main(args)