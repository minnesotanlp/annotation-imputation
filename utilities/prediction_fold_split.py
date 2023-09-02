# Generates the folds
import numpy as np
from tqdm import tqdm

# import from ../kernel_matrix_factorization
import sys
sys.path.append('../kernel_matrix_factorization')
from remove_fold import get_and_remove_fold
from make_path import make_path

# something like "SChem_{fold}_{datatype}.npy"
# must have 'fold' and 'datatype' in the format string
SplitFormatStr = str


def prediction_fold_split(annotations_location: str, texts_location: str, train_output: SplitFormatStr, val_output: SplitFormatStr, n_folds=5) -> None:
    for fold_index in tqdm(range(n_folds), desc="Folds", leave=False):
        # load the annotations and texts
        annotations = np.load(annotations_location, allow_pickle=True)
        texts = np.load(texts_location, allow_pickle=True)

        # get the train and val folds
        val_annotations, train_annotations = get_and_remove_fold(annotations, fold=fold_index, n_folds=n_folds)
        val_texts, train_texts = get_and_remove_fold(texts, fold=fold_index, n_folds=n_folds)

        assert len(train_annotations) == len(train_texts), f"{len(train_annotations)} != {len(train_texts)}"
        assert len(val_annotations) == len(val_texts), f"{len(val_annotations)} != {len(val_texts)}"

        # save the train and val folds
        datatype = "annotations"
        np.save(train_output.format(fold=fold_index, datatype=datatype), train_annotations)
        np.save(val_output.format(fold=fold_index, datatype=datatype), val_annotations)

        datatype = "texts"
        np.save(train_output.format(fold=fold_index, datatype=datatype), train_texts)
        np.save(val_output.format(fold=fold_index, datatype=datatype), val_texts)


if __name__ == '__main__':
    datasets = ["SChem", "ghc", "Sentiment", "politeness", "SChem5Labels", "Fixed2xSBIC"]

    for dataset in tqdm(datasets, desc="Datasets"):
        annotations_location = f"../datasets/cleaned/{dataset}_annotations.npy"
        texts_location = f"../datasets/cleaned/{dataset}_texts.npy"
        train_output = "../datasets/cleaned_folds/{dataset}/{dataset}_train_{fold}_{datatype}.npy".format(dataset=dataset, fold="{fold}", datatype="{datatype}")
        val_output = "../datasets/cleaned_folds/{dataset}/{dataset}_val_{fold}_{datatype}.npy".format(dataset=dataset, fold="{fold}", datatype="{datatype}")

        make_path(train_output)
        make_path(val_output)

        prediction_fold_split(annotations_location, texts_location, train_output, val_output)