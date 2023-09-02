'''
Given two KL divergence arrays, this script will compare them and compute
1. The mean of each KL divergence array
2. The standard deviation of each KL divergence array
3. How often the KL divergence of the first array is less than (better than) the KL divergence of the second array
It then saves the results to a .json file

(Primarily coded by Copilot, not GPT-4)
'''

import argparse
import numpy as np
import os

def make_path(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

def load_npy_file(file_path):
    return np.load(file_path, allow_pickle=True)

def compare_kl(kl1, kl2):
    max_len = min(len(kl1), len(kl2))
    if max_len < len(kl1):
        print(f"WARNING: The first KL divergence array is longer than the second KL divergence array. Only the last {max_len} elements will be used.")
    elif max_len < len(kl2):
        print(f"WARNING: The second KL divergence array is longer than the first KL divergence array. Only the last {max_len} elements will be used.")
    # grab the last max_len elements
    kl1 = kl1[-max_len:]
    kl2 = kl2[-max_len:]
    mean1 = np.mean(kl1)
    mean2 = np.mean(kl2)
    std1 = np.std(kl1)
    std2 = np.std(kl2)
    better = int(np.sum(kl1 < kl2))
    percent_better = better / len(kl1)
    worse = int(np.sum(kl1 > kl2))
    same = int(np.sum(kl1 == kl2))
    return mean1, mean2, std1, std2, better, percent_better, worse, same

def save_json_file(results, file_path):
    import json
    make_path(file_path)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two KL divergence arrays')
    parser.add_argument('--kl1', type=str, required=True, help='Path to the .npy file containing the first KL divergence array')
    parser.add_argument('--kl2', type=str, required=True, help='Path to the .npy file containing the second KL divergence array')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the .json file containing the results')
    parser.add_argument('--name1', type=str, default='1', help='Name of the first KL divergence array')
    parser.add_argument('--name2', type=str, default='2', help='Name of the second KL divergence array')

    args = parser.parse_args()

    kl1 = load_npy_file(args.kl1)
    kl2 = load_npy_file(args.kl2)

    mean1, mean2, std1, std2, better, percent_better, worse, same = compare_kl(kl1, kl2)

    results = {
        f"mean_{args.name1}": mean1,
        f"mean_{args.name2}": mean2,
        f"std_{args.name1}": std1,
        f"std_{args.name2}": std2,
        f"{args.name1}_better": better,
        f"{args.name1}_percent_better": percent_better,
        f"{args.name2}_better": worse,
        f"same": same
    }

    save_json_file(results, args.output_file)