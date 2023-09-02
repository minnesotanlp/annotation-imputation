import pandas as pd
import numpy as np
'''
In all of the npy dataset formats, missing ratings are represented by -1.
In all of the non-npy formats, missing ratings are represented by None (not NaN).
For the non-npy formats, note that the ID may not be an integer; it may be a string
'''

'''
Npy Annotations Format
Rows are examples, columns are users, values are ratings
(Essentially the same as matrix format, but no column names and no sentence_id/sentence columns.)
data.shape = (num_examples, num_users)
Usually saved as {dataset_name}_annotations.npy
'''
NpyAnnotationsFormat = np.ndarray

'''
Npy Texts Format
Each row has a single sentence.
(Essentially just the sentence column of matrix format.)
data.shape = (num_examples, 1)
Usually saved as {dataset_name}_texts.npy
'''
NpyTextsFormat = np.ndarray

'''
Original Format
Same as factor format in terms of the structure, but the column names may not exactly match.
To pick out the correct columns, use indexing rather than the names of the columns.
'''
OriginalDataFrame = pd.DataFrame

'''
Factor Format
Each rating is a rating from a single annotator for that example
Columns of CSV (order does not matter):
user_id, sentence_id, sentence1, label
(This is usually fixed with the `fix_columns` script.)
'''
FactorDataFrame = pd.DataFrame

'''
Matrix Format
first two columns are sentence_id followed by sentence1, columns after that are user_ids, 1 row per item (no duplicates), values are ratings/labels.
Missing ratings are represented by None or NaN.
'''
MatrixDataFrame = pd.DataFrame

'''
Aggregated Format
Each label is a aggregated/summarized (either through majority voting or averaging and rounding) rating
Essentially matrix format, but all the user_ids have been squashed into one label.
Columns of CSV:
sentence_id, sentence1, label
'''
AggregatedDataFrame = pd.DataFrame

'''
Missing Format
Each row represents a missing rating
Columns of CSV:
sentence_id, sentence1, user_id
'''
MissingDataFrame = pd.DataFrame