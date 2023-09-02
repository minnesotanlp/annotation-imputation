'''Doubles the value of the annotations for a npy dataset. Usually only used once to double the base dataset of a dataset like SBIC that has 0.5 as a rating'''
import numpy as np

def main(input_path: str, output_path: str, assert_int: bool = False):
    dataset = np.load(input_path, allow_pickle=True)
    # Double all the values except for -1s (which are empty ratings)
    dataset[dataset != -1] *= 2

    if assert_int:
        assert np.all(np.isclose(dataset, dataset.astype(int))), "The values are not integers after doubling"
        # Since the values are integers, we can convert them to integers
        dataset = dataset.astype(int)

    np.save(output_path, dataset, allow_pickle=True)
    print(f"Doubled dataset saved to {output_path}")
    print(f"Remember to duplicate the texts with the new dataset name as well (if needed).")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the npy file containing the annotations")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file for the train data")
    parser.add_argument("--assert_int", action="store_true", help="Whether to assert that the values are integers")
    args = parser.parse_args()
    main(args.input, args.output, args.assert_int)