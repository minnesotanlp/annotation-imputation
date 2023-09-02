'''
Take a folder of logs from running json on a bunch of models and extract the best RMSE, best factor, and best LR for each dataset.
'''
import os
import re
import json

def extract_data_from_txt(file_path):
    with open(file_path, "r") as f:
        file_content = f.read()
    
    best_rmse = float(re.search(r"Best RMSE = (\d+\.\d+)", file_content).group(1))
    best_factor = int(re.search(r"Best Factor: (\d+)", file_content).group(1))
    best_lr = float(re.search(r"Best LR: (\d+\.\d+|\d+e-\d+)", file_content).group(1))
    
    return {"best_rmse": best_rmse, "best_factor": best_factor, "best_lr": best_lr}

def process_folders(folder_path):
    results = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                dataset_name = os.path.basename(root)
                if dataset_name not in results:
                    results[dataset_name] = {}
                results[dataset_name].update(extract_data_from_txt(file_path))
    assert results, f"Could not find any files or results in the folder path '{folder_path}'"
    return results

def save_results_to_json(results, output_file_path):
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=4)

def main(folder_path, output_file_path):
    results = process_folders(folder_path)
    save_results_to_json(results, output_file_path)

if __name__ == "__main__":
    folder_path = "../datasets/disagreement_datasets/clean-ncf/ncf_logs"  # Replace with the path to your folder
    output_file_path = "ncf_results.json"  # Replace with the desired output file path
    main(folder_path, output_file_path)