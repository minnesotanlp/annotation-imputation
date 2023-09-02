import json
import matplotlib.pyplot as plt
import numpy as np
import os

def make_path(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

# Load the JSON data
with open("outputs/quick_test_disagreement_analysis_overall.json", "r") as f:
    data = json.load(f)

# Function to create bar chart
def create_bar_chart(dataset_name, metric_name):
    dataset = data[dataset_name]

    # Extract values for low, medium, and high disagreement
    low_values = [dataset["low_disagreement"][group][metric_name] for group in ["original", "imputed"]]
    medium_values = [dataset["medium_disagreement"][group][metric_name] for group in ["original", "imputed"]]
    high_values = [dataset["high_disagreement"][group][metric_name] for group in ["original", "imputed"]]

    # Set up the bar chart
    labels = ["Original", "Imputed"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, low_values, width, label="Low Disagreement")
    rects2 = ax.bar(x, medium_values, width, label="Medium Disagreement")
    rects3 = ax.bar(x + width, high_values, width, label="High Disagreement")

    # Add labels, title, and custom x-axis tick labels
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"{dataset_name} {metric_name.replace('_', ' ').title()} by Disagreement Level")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(), 2)
            ax.annotate(f"{height}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom")
            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # make the y-axis slightly taller to make room for the labels
    plt.ylim(0, 1.1 * max(low_values + medium_values + high_values))

    # Save the bar chart
    save_location = f"graphs/{dataset_name}_{metric_name}_by_disagreement_level.png"
    make_path(save_location)
    plt.savefig(save_location)

    # Show the bar chart
    plt.show()

# Create bar charts for each dataset and metric
for dataset_name in data.keys():
    for metric_name in ["weighted_f1_mean", "accuracy_mean"]:
        create_bar_chart(dataset_name, metric_name)