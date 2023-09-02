import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
import pandas as pd
from dataset_formats import NpyAnnotationsFormat, NpyTextsFormat, FactorDataFrame
from dataset_format_conversions import factor_to_npy_format, npy_to_factor_format, factor_to_npy_format_with_npy_reference
from annotation_masks import boolean_annotation_mask, BooleanAnnotationMask, apply_boolean_mask_to_annotations
from simple_train_val_test_split import simple_train_val_test_split

DEFAULT_TRAIN_SPLIT = 0.9
DEFAULT_VAL_SPLIT = 0.05

def npy_to_npy_one_anno_per_cross_split(annotations: NpyAnnotationsFormat, texts: NpyTextsFormat, train_split=0.9, val_split=0.05, seed: Optional[int]=None) -> Tuple[NpyAnnotationsFormat, NpyTextsFormat, NpyAnnotationsFormat, NpyTextsFormat, NpyAnnotationsFormat, NpyTextsFormat]:
    train_factor_df, validation_factor_df, test_factor_df = npy_to_factor_one_anno_per_cross_split(annotations, texts, train_split=train_split, val_split=val_split, seed=seed)

    print("Converting back to npy format...")
    train_annotations, train_texts = factor_to_npy_format_with_npy_reference(train_factor_df, annotations, texts)
    validation_annotations, validation_texts = factor_to_npy_format_with_npy_reference(validation_factor_df, annotations, texts)
    test_annotations, test_texts = factor_to_npy_format_with_npy_reference(test_factor_df, annotations, texts)
    print("Conversion back to npy format complete.")

    assert annotations.shape == train_annotations.shape == validation_annotations.shape == test_annotations.shape, f"At least one of the shapes does not match the others: {annotations.shape}, {train_annotations.shape}, {validation_annotations.shape}, {test_annotations.shape}"

    return train_annotations, train_texts, validation_annotations, validation_texts, test_annotations, test_texts

def npy_to_factor_one_anno_per_cross_split(annotations: NpyAnnotationsFormat, texts: NpyTextsFormat, train_split=0.9, val_split: Optional[float]=0.05, seed: Optional[int]=None) -> Tuple[FactorDataFrame, FactorDataFrame, FactorDataFrame]:
    '''Splits the given npy dataset into a train, validation, and test set in factor format. If val_split is None, then train will contain as few examples as possible while still retaining one per row and column, validation will be empty, and test will contain the rest. The test split is implied by the given train and validation split.'''
    # how much should be split off for both val and test combined
    if val_split is None:
        val_and_test_split = None
    else:
        val_and_test_split = 1 - train_split

    # get the boolean masks for the annotations
    train_mask, val_and_test_mask = get_npy_one_anno_masks(annotations, val_split=val_and_test_split, seed=seed)

    # apply the masks to the annotations
    train_annotations = apply_boolean_mask_to_annotations(annotations, train_mask)
    val_and_test_annotations = apply_boolean_mask_to_annotations(annotations, val_and_test_mask)

    # convert back to a factor dataframe
    train_factor_df = npy_to_factor_format(train_annotations, texts)
    val_and_test_factor_df = npy_to_factor_format(val_and_test_annotations, texts)

    # split the factor dataframe into validation and test sets
    simple_val_split = (val_split / val_and_test_split) if val_and_test_split not in (None, 0) else 0
    _empty_train_factor_df, validation_factor_df, test_factor_df = simple_train_val_test_split(val_and_test_factor_df, train_split=0, val_split=simple_val_split)

    return train_factor_df, validation_factor_df, test_factor_df

def factor_to_factor_one_anno_per_cross_split(factor_df: FactorDataFrame, train_split=DEFAULT_TRAIN_SPLIT, val_split=DEFAULT_VAL_SPLIT, seed: Optional[int]=None) -> Tuple[FactorDataFrame, FactorDataFrame, FactorDataFrame]:
    '''Generate a train/val/test split where train has at least one annotation per row and column. Test split is implied by the other two.'''
    # first, convert the factor dataframe to numpy annotations
    print("Converting to npy format...")
    annotations, texts = factor_to_npy_format(factor_df)
    print("Conversion to npy format complete.")

    train_factor_df, validation_factor_df, test_factor_df = npy_to_factor_one_anno_per_cross_split(annotations, texts, train_split=train_split, val_split=val_split, seed=seed)

    # for sanity checking, make sure the lengths of the dfs add up to the original
    assert len(train_factor_df) + len(validation_factor_df) + len(test_factor_df) == len(factor_df), f"{len(train_factor_df)} + {len(validation_factor_df)} + {len(test_factor_df)} != {len(factor_df)}"

    return train_factor_df, validation_factor_df, test_factor_df

def get_npy_one_anno_masks(annotations: NpyAnnotationsFormat, val_split: Optional[float]=0.1, seed: Optional[int]=None) -> Tuple[BooleanAnnotationMask, BooleanAnnotationMask]:
    '''
    Pick out one annotation per row and column, and then pick out extras until there are only val_split left. If val_split is None, don't pick out any extras.
    Returns two boolean masks, one for the chosen annotations and one for the unchosen (but not missing) annotations.
    '''
    if seed is not None:
        np.random.seed(seed)

    annotations = annotations.copy()
    boolean_mask: BooleanAnnotationMask = boolean_annotation_mask(annotations)
    # how many annotations there are in total
    num_annotations = np.sum(boolean_mask)
    # how many annotations should be split off for validation
    if val_split is None:
        num_validation = None
    else:
        num_validation = round(num_annotations * val_split)
    m, n = annotations.shape

    # generate a list of all the indices of the annotations
    indices = np.argwhere(boolean_mask)

    # ensure that there's at least one annotation per row and column
    if np.sum(boolean_mask, axis=1).min() < 1:
        raise ValueError("There are rows with no annotations")
    
    if np.sum(boolean_mask, axis=0).min() < 1:
        raise ValueError("There are columns with no annotations")
    
    rows = np.arange(m)
    cols = np.arange(n)

    assert len(rows) == m

    # shuffle the indices
    np.random.shuffle(indices)

    # pick out one annotation per row and column
    chosen_indices = []
    # create tqdm bar not linked to loop
    pbar = tqdm(total=m + n, desc="One anno per cross", leave=False)
    for i, j in indices:
        if i in rows or j in cols:
            chosen_indices.append((i, j))
            if i in rows:
                rows = np.setdiff1d(rows, i)
                pbar.update(1)
            if j in cols:
                cols = np.setdiff1d(cols, j)
                pbar.update(1)
            
            if len(rows) == len(cols) == 0:
                break

    # make sure the rows and cols are empty
    assert len(rows) == 0, len(rows)
    assert len(cols) == 0, cols
        
    # create a list of the indices that weren't chosen
    unchosen_indices = list(set(map(tuple, indices)) - set(map(tuple, chosen_indices)))
    assert len(unchosen_indices) == len(indices) - len(chosen_indices), f"{len(unchosen_indices)} != {len(indices)} - {len(chosen_indices)}"

    # how many annotations have been chosen
    num_current = len(chosen_indices)
    # how many annotations are needed for training
    if val_split is None:
        num_train = num_current
    else:
        num_train = num_annotations - num_validation
    num_needed = int(num_train - num_current)
    
    assert num_needed == 0 or val_split is not None, f"{num_needed} != 0 and {val_split} is None"

    # since the indices are shuffled, we can just take the first num_needed
    extra_indices = unchosen_indices[:num_needed]
    num_extra = len(extra_indices)
    assert len(extra_indices) + num_current == num_train, f"{num_extra} + {num_current} != {num_train}"
    # remove the extra indices from the unchosen indices
    unchosen_indices = unchosen_indices[num_needed:]

    # add the extra indices to the chosen indices
    chosen_indices.extend(extra_indices)

    assert len(chosen_indices) == num_train, f"{len(chosen_indices)} != {num_train}"

    # create a mask of the chosen indices
    chosen_mask: BooleanAnnotationMask = np.zeros_like(boolean_mask)
    if len(chosen_indices) > 0:
        chosen_mask[tuple(zip(*chosen_indices))] = True

    assert np.sum(chosen_mask) == len(chosen_indices), f"{np.sum(chosen_mask)} != {len(chosen_indices)}"

    # create a mask of the unchosen indices
    unchosen_mask: BooleanAnnotationMask = np.zeros_like(boolean_mask)
    if len(unchosen_indices) > 0:
        unchosen_mask[tuple(zip(*unchosen_indices))] = True

    if val_split is not None:
        assert np.all((unchosen_mask + chosen_mask) == boolean_mask), f"The chosen and not chosen masks together don't equal the original mask. We chose {int(np.sum(chosen_mask))} and didn't choose {int(np.sum(unchosen_mask))} annotations, which together is {int(np.sum(chosen_mask) + np.sum(unchosen_mask))} annotations. However, the original mask has {int(np.sum(boolean_mask))} annotations. If the math seems right, then the positions of the annotations in the masks is what's triggering this error."

    return chosen_mask, unchosen_mask

if __name__ == "__main__":
    dataset_name = 'SChem'
    annotations = np.load(f'../datasets/cleaned/{dataset_name}_annotations.npy', allow_pickle=True)
    texts = np.load(f'../datasets/cleaned/{dataset_name}_texts.npy', allow_pickle=True)

    # annotations = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [-1, -1, -1, -1, 1], [-1, -1, -1, -1, 1]])
    # texts = np.array(["a", "b", "c", "d"])

    print(f"Dataset shape: {annotations.shape}")

    # convert to factor format
    factor_df = npy_to_factor_format(annotations, texts)

    train, val, test = factor_to_factor_one_anno_per_cross_split(factor_df, val_split=None, seed=60)
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")