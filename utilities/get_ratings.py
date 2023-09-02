import numpy as np

# dataset_location = "../datasets/cleaned/politeness_annotations.npy"
dataset_location = "../datasets/cleaned/Sentiment_annotations.npy"

data = np.load(dataset_location, allow_pickle=True)

labels = np.unique(data)
labels = labels[labels != -1]

print(labels)
print(len(labels))

print("All labels")
print(np.unique(data))
print(len(np.unique(data)))