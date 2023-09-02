import numpy as np

import sys
sys.path.append('..')
from dataset_formats import NpyAnnotationsFormat

# usually used to indicate where annotations are present/missing, respectively
# True/False values
BooleanAnnotationMask = np.ndarray
# 1/0 values
BinaryAnnotationMask = np.ndarray

def boolean_annotation_mask(annotations: NpyAnnotationsFormat, in_place: bool=False) -> BooleanAnnotationMask:
    '''Converts annotations to binary matrix, where True represents an annotation and False represents a missing annotation.'''
    if not in_place:
        annotations = annotations.copy()
    annotations[annotations != -1] = True
    annotations[annotations == -1] = False
    return annotations

def binary_annotation_mask(annotations: NpyAnnotationsFormat, in_place: bool=False) -> BinaryAnnotationMask:
    '''Converts annotations to binary matrix, where 1 represents an annotation and 0 represents a missing annotation.'''
    if not in_place:
        annotations = annotations.copy()
    annotations[annotations != -1] = 1
    annotations[annotations == -1] = 0
    return annotations

def binary_to_boolean_mask(binary_mask: BinaryAnnotationMask) -> BooleanAnnotationMask:
    '''Converts a binary mask to a boolean mask.'''
    return binary_mask.astype(bool)

def boolean_to_binary_mask(boolean_mask: BooleanAnnotationMask) -> BinaryAnnotationMask:
    '''Converts a boolean mask to a binary mask.'''
    return boolean_mask.astype(np.float32)

def apply_boolean_mask_to_annotations(annotations: NpyAnnotationsFormat, boolean_mask: BooleanAnnotationMask, inplace=False) -> NpyAnnotationsFormat:
    '''Given annotations and a boolean mask, returns a copy of the annotations with the mask applied.'''
    if not inplace:
        annotations = annotations.copy()
    annotations[boolean_mask == False] = -1
    return annotations

def apply_binary_mask_to_annotations(annotations: NpyAnnotationsFormat, binary_mask: BinaryAnnotationMask, inplace=False) -> NpyAnnotationsFormat:
    '''Given annotations and a boolean mask, returns a copy of the annotations with the mask applied.'''
    if not inplace:
        annotations = annotations.copy()
    annotations[binary_mask == 0] = -1
    return annotations

def get_boolean_missing_annotation_mask(annotations: NpyAnnotationsFormat) -> BooleanAnnotationMask:
    '''Returns a boolean mask where True indicates a missing annotation and False indicates a present annotation.'''
    return annotations == -1