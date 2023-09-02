'''Given a npy file, generate folds such that if any of the k folds is removed from the dataset, there's at least one annotation per row and column left in the dataset (and usually a bit more than that).'''

import numpy as np
from tqdm import tqdm
from typing import List

# import from ../kernel_matrix_factorization
import sys
sys.path.append('../../kernel_matrix_factorization')
from remove_fold import remove_fold
from dataset_formats import NpyAnnotationsFormat, NpyTextsFormat
from one_anno_per_cross_split import factor_to_factor_one_anno_per_cross_split
from annotation_masks import BooleanAnnotationMask, BinaryAnnotationMask, binary_to_boolean_mask, get_boolean_missing_annotation_mask, boolean_to_binary_mask
from dataset_format_conversions import npy_to_factor_format, factor_to_npy_format

class NotEnoughAnnotationsError(ValueError):
    pass

def main(annotations: NpyAnnotationsFormat, texts: NpyTextsFormat, n_folds=5, train_split: float=0.9, val_split: float=0.1) -> List[BooleanAnnotationMask]:
    # # if there's not at least two annotations per column, then raise an error
    # if np.any(np.sum(data, axis=0) < 2):
    #     raise NotEnoughAnnotationsError("There are columns with less than two annotations")
    
    # first test to see if the standard split works
    data_per_fold = len(annotations) // n_folds
    train_annotation_folds = [remove_fold(annotations, i, n_folds) for i in range(n_folds)]
    train_text_folds = [remove_fold(texts, i, n_folds) for i in range(n_folds)]
    failed_folds = []
    for i in tqdm(range(n_folds), desc="Easy Folds", leave=False):
        train_annotations = train_annotation_folds[i]
        train_texts = train_text_folds[i]
        print("Converting to factor format...")
        factor_format_train_data = npy_to_factor_format(train_annotations, train_texts)
        print("Conversion complete!")
        try:
            factor_to_factor_one_anno_per_cross_split(factor_format_train_data, train_split=train_split, val_split=val_split)
        except ValueError:
            failed_folds.append(i)

    if not failed_folds:
        # generate the boolean masks for the folds
        # the boolean masks should only include actual (non -1) annotations
        ans = []
        # get mask that is True if and only if the annotation is missing
        missing_mask: BooleanAnnotationMask = get_boolean_missing_annotation_mask(annotations)
        for i in tqdm(range(n_folds), desc="Generating Folds", leave=False):
            # create a mask where all of the rows for the fold are filled with True
            # start with a mask of all 0s
            binary_mask = np.zeros(annotations.shape, dtype=bool)
            # set the rows for the fold to 1
            binary_mask[i * data_per_fold:(i + 1) * data_per_fold] = 1
            # create a boolean mask from the binary mask
            boolean_mask = binary_to_boolean_mask(binary_mask)
            # combine the boolean mask with the missing mask
            # should be True if and only if the annotation is not missing and is in the fold
            boolean_mask = boolean_mask & ~missing_mask
            ans.append(boolean_mask)
        return ans
    
if __name__ == "__main__":
    # dataset = "SChem5Labels"
    # data_file = f"../datasets/cleaned/{dataset}_annotations.npy"
    # texts_file = f"../datasets/cleaned/{dataset}_texts.npy"
    # annotations = np.load(data_file, allow_pickle=True)
    # texts = np.load(texts_file, allow_pickle=True)
    # ans = main(annotations, texts)
    # print(len(ans))
    # assert ans

    dataset = "SChem5Labels"
    data_file = f"../datasets/cleaned_folds/{dataset}/{dataset}_train_4_annotations.npy"
    texts_file = f"../datasets/cleaned_folds/{dataset}/{dataset}_train_4_texts.npy"
    annotations = np.load(data_file, allow_pickle=True)
    texts = np.load(texts_file, allow_pickle=True)

    # this passes, so I have no idea why NCF was failing
    factor_format_train_data = npy_to_factor_format(annotations, texts)
    factor_to_factor_one_anno_per_cross_split(factor_format_train_data, train_split=0.9, val_split=0.1)