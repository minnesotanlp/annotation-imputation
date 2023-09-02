'''
Given a directory where the only json files in it are log files from kernel matrix factorization, this script will extract the best model's validation and test RMSE for each dataset and save it to a json file.
'''

import os
import json
import re

def extract_data_from_json(file_path):
    VAL_RMSE_KEY = "val_rmse"
    TEST_RMSE_KEY = "test_rmse"

    # load the json file
    with open(file_path, "r") as f:
        file_content = json.load(f)

    # extract the best model
    best_model = file_content["best_model"]

    # extract the validation and test RMSE
    val_rmse = file_content[best_model][VAL_RMSE_KEY]
    test_rmse = file_content[best_model][TEST_RMSE_KEY]
    
    return {VAL_RMSE_KEY: val_rmse, TEST_RMSE_KEY: test_rmse}

def process_folders(folder_path):
    results = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                dataset_name = os.path.basename(file).split("_")[0]
                if dataset_name not in results:
                    results[dataset_name] = {}
                results[dataset_name].update(extract_data_from_json(file_path))
    assert results, f"Could not find any files or results in the folder path '{folder_path}'"
    return results

def save_results_to_json(results, output_file_path):
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=4)

def main(folder_path, output_file_path):
    results = process_folders(folder_path)
    save_results_to_json(results, output_file_path)

if __name__ == "__main__":
    folder_path = "../datasets/disagreement_datasets/kernel/run_data"  # Replace with the path to your folder
    output_file_path = "kernel_results.json"  # Replace with the desired output file path
    main(folder_path, output_file_path)