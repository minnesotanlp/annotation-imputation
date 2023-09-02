import numpy as np
from label_distribution import label_distribution
import os

def make_path(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

def main(args):
    orig = np.load(args.orig, allow_pickle=True)
    imputed = np.load(args.imputed, allow_pickle=True)

    assert orig.shape[1] == imputed.shape[1], f"The number of annotators in the original and imputed arrays must be the same. Got {orig.shape[1]} and {imputed.shape[1]}."

    if orig.shape[0] > imputed.shape[0]:
        difference = orig.shape[0] - imputed.shape[0]
        print(f"The number of items in the original and imputed arrays is different. Got {orig.shape[0]} and {imputed.shape[0]}. We will assume that the imputed array is missing the first {difference} rows.")
        orig = orig[difference:, :]

    assert orig.shape == imputed.shape, f"The shape of the original and imputed arrays must be the same at this point. Got {orig.shape} (original) and {imputed.shape} (imputed). Did you accidentally pass in the imputed array as the original array?"

    if args.double:
        orig = orig * 2
        imputed = imputed * 2

    # make sure that the original data is already rounded to the nearest integer
    assert np.all(np.isclose(orig, np.round(orig))), "The original data must be rounded to the nearest integer. Are you dealing with the SBIC dataset? If so, use the `--double` flag."

    # round the imputed values to the nearest integer
    imputed = np.round(imputed)

    unique_labels = sorted(list(np.unique(orig)))
    unique_labels.remove(-1) # remove the -1 label (which is used to indicate missing values)

    # create distributions for each row in the original and imputed arrays
    orig_distribution = label_distribution(unique_labels, orig)
    assert orig_distribution.shape == (orig.shape[0], len(unique_labels)), f"The shape of the original distribution should be {orig.shape[0], len(unique_labels)}. Got {orig_distribution.shape} instead."

    imputed_distribution = label_distribution(unique_labels, imputed)
    assert orig_distribution.shape == imputed_distribution.shape, f"The shape of the original and imputed distributions must be the same. Got {orig_distribution.shape} and {imputed_distribution.shape}."

    # save the distributions
    make_path(args.orig_distribution_output)
    make_path(args.imputed_distribution_output)
    np.save(args.orig_distribution_output, orig_distribution, allow_pickle=True)
    np.save(args.imputed_distribution_output, imputed_distribution, allow_pickle=True)

    # compute the KL divergence between the distributions
    epsilon = 1e-10
    kl_divergences = np.sum(orig_distribution * np.log((orig_distribution + epsilon) / (imputed_distribution + epsilon)), axis=1)

    assert kl_divergences.shape == (orig.shape[0],), f"The shape of the KL divergences should be {orig.shape[0]}. Got {kl_divergences.shape} instead."

    # save the KL divergences
    make_path(args.kl_output)
    np.save(args.kl_output, kl_divergences, allow_pickle=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orig",
        type=str,
        required=True,
        help="Location of the original annotations file."
    )
    parser.add_argument(
        "--imputed",
        type=str,
        required=True,
        help="Location of the imputed annotations file."
    )
    parser.add_argument(
        "--orig_distribution_output",
        type=str,
        required=True,
        help="Location to save the original distribution file."
    )
    parser.add_argument(
        "--imputed_distribution_output",
        type=str,
        required=True,
        help="Location to save the imputed distribution file."
    )
    parser.add_argument(
        "--kl_output",
        type=str,
        required=True,
        help="Location to save the KL divergences file."
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="Whether to double the datasets. This is useful for the SBIC dataset, where 0.5 is one of the labels."
    )
    args = parser.parse_args()
    main(args)