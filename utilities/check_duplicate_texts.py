import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='npy texts file to check for duplicates', required=True)

    # args = parser.parse_args(['--file', 'datasets/SChem5Labels_texts.npy'])
    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        texts = np.load(f, allow_pickle=True)
        assert len(texts.shape) == 2, f"Texts should be a 2D array with shape (n, 1). Instead, it has shape {texts.shape}"
        assert texts.shape[1] == 1, f"Texts should be a 2D array with shape (n, 1). Instead, it has shape {texts.shape}"
        
        # get duplicate texts
        unique_texts, counts = np.unique(texts, return_counts=True)

        # get the indices of the duplicate texts
        duplicate_indices = np.where(counts > 1)[0]

        # get the duplicate texts
        duplicate_texts = unique_texts[duplicate_indices]

        # print the duplicate texts alongside their indices and counts
        print(f"The indicies, texts, and counts of duplicates:\n{np.stack((duplicate_indices, duplicate_texts, counts[duplicate_indices])).T}")
        print("Done checking for duplicates")

if __name__ == '__main__':
    main()