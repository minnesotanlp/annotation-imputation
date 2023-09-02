from typing import Union
import numpy as np
from dataset_formats import NpyAnnotationsFormat, NpyTextsFormat

DEFAULT_FOLD = 0
DEFAULT_N_FOLDS = 5

def get_fold_indices(n_examples, fold=DEFAULT_FOLD, n_folds=DEFAULT_N_FOLDS):
    '''
    Get the indices of the nth fold.
    '''
    fold_size = n_examples // n_folds
    fold_start = fold_size * fold
    fold_end = fold_start + fold_size

    return fold_start, fold_end

def get_fold(npy_data: Union[NpyAnnotationsFormat, NpyTextsFormat], fold=DEFAULT_FOLD, n_folds=DEFAULT_N_FOLDS):
    '''
    Get the annotations and texts corresponding to the nth fold.
    '''
    n_examples = len(npy_data)

    fold_start, fold_end = get_fold_indices(n_examples, fold=fold, n_folds=n_folds)

    # get the data corresponding to the nth fold
    npy_data = npy_data[fold_start:fold_end]

    return npy_data

def get_and_remove_fold(npy_data: Union[NpyAnnotationsFormat, NpyTextsFormat], fold=DEFAULT_FOLD, n_folds=DEFAULT_N_FOLDS):
    n_examples = len(npy_data)

    fold_start, fold_end = get_fold_indices(n_examples, fold=fold, n_folds=n_folds)

    # get the data corresponding to the nth fold
    npy_data_fold = npy_data[fold_start:fold_end]

    # remove the data corresponding to the nth fold
    npy_data_no_fold = np.concatenate((npy_data[:fold_start], npy_data[fold_end:]), axis=0)

    return npy_data_fold, npy_data_no_fold

def remove_fold(npy_data: Union[NpyAnnotationsFormat, NpyTextsFormat], fold=DEFAULT_FOLD, n_folds=DEFAULT_N_FOLDS):
    '''
    Remove the annotations and texts corresponding to the nth fold.
    '''
    npy_data = npy_data.copy()

    n_examples = len(npy_data)

    fold_start, fold_end = get_fold_indices(n_examples, fold=fold, n_folds=n_folds)

    # remove the data corresponding to the nth fold
    npy_data = np.concatenate((npy_data[:fold_start], npy_data[fold_end:]), axis=0)

    return npy_data

def remove_fold_both(annotation: NpyAnnotationsFormat, texts: NpyTextsFormat, fold=DEFAULT_FOLD, n_folds=DEFAULT_N_FOLDS):
    '''
    Remove the annotations and texts corresponding to the nth fold.
    '''
    return remove_fold(annotation, fold=fold, n_folds=n_folds), remove_fold(texts, fold=fold, n_folds=n_folds)