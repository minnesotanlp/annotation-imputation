import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# a string to a location with {dataset_name} as a placeholder for the dataset name
FormatStr = str

def plot_metrics_for_all_datasets(dataset_names, results_format_str: FormatStr, display=False):
    '''Plots the results. Will save the plot to the same folder that the results are in.'''
    all_data = {}
    for dataset_name in tqdm(dataset_names, desc="Loading results"):
        # CHANGE THIS PATH TO WHEREVER YOU STORE THE RESULTS
        file_path = results_format_str.format(dataset_name=dataset_name)
        with open(file_path) as f:
            all_data[dataset_name] = json.load(f)

    # create the output 'graphs' folder
    output_folder = os.path.join(os.path.dirname(results_format_str), "graphs")
    os.makedirs(output_folder, exist_ok=True)

    methods = list(all_data[dataset_names[0]].keys())
    n_methods = len(methods)
    n_datasets = len(dataset_names)

    rmse_scores = [[all_data[dataset_name][method]["rmse"] for dataset_name in dataset_names] for method in methods]
    weighted_f1 = [[all_data[dataset_name][method]["classification_report"]["weighted avg"]["f1-score"] for dataset_name in dataset_names] for method in methods]
    accuracy = [[all_data[dataset_name][method]["classification_report"]["accuracy"] for dataset_name in dataset_names] for method in methods]

    custom_ordering = list(range(n_methods))
    # In case you want to change the order of the methods
    custom_ordering = [0, 1, 4, 2, 3]

    assert len(custom_ordering) == n_methods

    rmse_scores = [rmse_scores[i] for i in custom_ordering]
    weighted_f1 = [weighted_f1[i] for i in custom_ordering]
    accuracy = [accuracy[i] for i in custom_ordering]

    methods = [methods[i] for i in custom_ordering]

    def annotate_bars(bars):
        for bar in bars:
            height = bar.get_height()
            plt.annotate('{:.2f}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        rotation='vertical')

    def plot_grouped_bar_graph(title, all_values, save_path):
        plt.figure()
        bar_width = 0.15
        n_bars = len(all_values)

        for i in range(n_bars):
            x_positions = np.arange(n_datasets) + i * bar_width
            bars = plt.bar(x_positions, all_values[i], width=bar_width, label=methods[i])
            annotate_bars(bars)

        plt.title(title)
        plt.xlabel("Datasets")
        plt.ylabel("Score")
        plt.xticks(np.arange(n_datasets) + (n_bars - 1) * bar_width / 2, dataset_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        if display:
            plt.show()

    plot_grouped_bar_graph("RMSE Comparison", rmse_scores, os.path.join(output_folder, "RMSE_comparison.png"))
    plot_grouped_bar_graph("Weighted F1 Comparison", weighted_f1, os.path.join(output_folder, "Weighted_F1_comparison.png"))
    plot_grouped_bar_graph("Accuracy Comparison", accuracy, os.path.join(output_folder, "Accuracy_comparison.png"))

if __name__ == "__main__":
    # Example usage:
    # All datasets:
    # datasets = ["SChem", "SChem5Labels", "ghc", "Fixed2xSBIC", "Sentiment", "politeness"]
    # Just one dataset:
    datasets = ["SChem"]
    results_format_str = "../../datasets/rmse_test/results/{dataset_name}_results.json"
    plot_metrics_for_all_datasets(datasets, results_format_str, display=True)