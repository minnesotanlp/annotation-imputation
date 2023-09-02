from typing import Tuple

from dataset_formats import NpyAnnotationsFormat, NpyTextsFormat, MatrixDataFrame, FactorDataFrame

from factor_to_matrix_form import factor_to_matrix_form, factor_to_matrix_form_with_reference
from matrix_to_factor_form import matrix_to_factor_form
from matrix_to_npy_format import matrix_to_npy_format
from npy_to_matrix_format import npy_to_matrix_format

def npy_to_factor_format(annotations: NpyAnnotationsFormat, texts: NpyTextsFormat) -> FactorDataFrame:
    matrix_df = npy_to_matrix_format(annotations, texts)
    factor_df = matrix_to_factor_form(matrix_df)
    return factor_df

def factor_to_npy_format(factor_df: FactorDataFrame) -> Tuple[NpyAnnotationsFormat, NpyTextsFormat]:
    matrix_df = factor_to_matrix_form(factor_df)
    annotations, texts = matrix_to_npy_format(matrix_df)
    return annotations, texts

def factor_to_npy_format_with_matrix_reference(factor_df: FactorDataFrame, reference_df: MatrixDataFrame) -> Tuple[NpyAnnotationsFormat, NpyTextsFormat]:
    matrix_df = factor_to_matrix_form_with_reference(factor_df, reference_df)
    annotations, texts = matrix_to_npy_format(matrix_df)

    assert matrix_to_npy_format(reference_df)[0].shape == annotations.shape, f"{matrix_to_npy_format(reference_df)[0].shape} != {annotations.shape}"
    assert matrix_to_npy_format(reference_df)[1].shape == texts.shape, f"{matrix_to_npy_format(reference_df)[1].shape} != {texts.shape}"
    return annotations, texts

def factor_to_npy_format_with_npy_reference(factor_df: FactorDataFrame, reference_annotations: NpyAnnotationsFormat, reference_texts: NpyTextsFormat) -> Tuple[NpyAnnotationsFormat, NpyTextsFormat]:
    reference_df = npy_to_matrix_format(reference_annotations, reference_texts)
    return factor_to_npy_format_with_matrix_reference(factor_df, reference_df)