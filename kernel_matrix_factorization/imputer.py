import pandas as pd
from typing import Tuple, Optional
from dataset_formats import MatrixDataFrame, NpyAnnotationsFormat, NpyTextsFormat
from dataset_format_conversions import npy_to_matrix_format, matrix_to_npy_format

# string that can be loaded into json
# usually the result of json.dumps
JSONstr = str

class Imputer:
    def __init__(self, name: str):
        self.name = name

    def impute(self, df: MatrixDataFrame) -> Tuple[MatrixDataFrame, JSONstr]:
        '''
        Impute the given matrix and return the imputed matrix and any extra data from the imputation.
        May modify the given matrix.
        '''
        raise NotImplementedError
    
class NpyImputer:
    def __init__(self, name: str):
        self.name = name

    def impute(self, annotations: Optional[NpyAnnotationsFormat], texts: Optional[NpyTextsFormat]) -> Tuple[NpyAnnotationsFormat, JSONstr]:
        '''
        Impute the given matrix and return the imputed matrix and any extra data from the imputation.
        May modify the given matrix.
        '''
        raise NotImplementedError

def impute_matrix_with_npy_imputer(npy_imputer: NpyImputer, df: MatrixDataFrame):
    '''
    Impute the given matrix with the given NpyImputer.
    '''
    annotations, texts = matrix_to_npy_format(df)
    imputed_annotations, extra_data = npy_imputer.impute(annotations, texts)
    imputed_df = npy_to_matrix_format(imputed_annotations, texts)
    return imputed_df, extra_data

class NpyImputerWrapper(Imputer):
    def __init__(self, npy_imputer: NpyImputer, name: Optional[str]=None):
        if name is None:
            name = npy_imputer.name
        super().__init__(name)
        self.npy_imputer = npy_imputer

    def impute(self, df: MatrixDataFrame) -> Tuple[MatrixDataFrame, JSONstr]:
        '''
        Impute the given matrix and return the imputed matrix and any extra data from the imputation.
        May modify the given matrix.
        '''
        return impute_matrix_with_npy_imputer(self.npy_imputer, df)