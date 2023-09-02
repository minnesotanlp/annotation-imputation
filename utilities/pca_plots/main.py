import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def main(annotations_file, name):
    # Read in the annotations file
    annotations = np.load(annotations_file, allow_pickle=True)
    # replace all the -1s with 10
    annotations[annotations == -1] = 10
    pca = PCA(2) 
    df = pca.fit_transform(annotations)
    kmeans = KMeans(n_clusters= 1)
    label = kmeans.fit_predict(df)
    u_labels = np.unique(label)
    for i in u_labels:
        plt.scatter(df[label == i , 0] , df[label == i , 1], label = i)
    # save it
    plt.savefig(name + ".png")
    plt.title(name)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_file", type=str)
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    main(args.annotations_file, args.name)