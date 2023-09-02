import numpy as np
import pandas as pd
import sys
import torch.utils.data as data
from typing import Tuple, Dict

sys.path.append("../kernel_matrix_factorization")
from annotation_masks import BooleanAnnotationMask, get_boolean_missing_annotation_mask
from dataset_formats import NpyAnnotationsFormat
from one_anno_per_cross_split import get_npy_one_anno_masks
from remove_fold import remove_fold

def boolean_mask_to_data_label(annotations: NpyAnnotationsFormat, matrix: BooleanAnnotationMask) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    '''Given annotations and a boolean mask over those annotations, return the data and label dictionaries.
    '''
    indices = matrix.nonzero()
    data = {
        "item": indices[0],
        "user": indices[1],
    }
    label = {
        "rating": annotations[indices],
    }

    return data, label

def load_all_ghc(args, test_num=100):
    """We load all the three file here to save time in each epoch."""
    # load the data
    input_anno = np.load(args.input_path)

    # take out the fold
    fold_index = args.fold
    n_folds = args.n_folds
    if fold_index != -1:
        input_anno = remove_fold(input_anno, fold=fold_index, n_folds=n_folds)

    # [London] The original code from Jaehyung used 80% for train and val (90%/10% split within the 80%), and 20% for test.
    # [London] I'm changing this to match Kernel: 90% for train, 5% for val, and 5% for test.
    # [London] I do think it's weird that we never seem to use the validation data. If this is the case, I think we should do 0% val and 10% test.
    # [London] However, I don't have the time to implement/check that right now.

    # [Old Code]
    # n_train, n_test = int(0.8 * 0.9 * n_samples), int(0.2 * n_samples)
    # [New Code]
    train_frac = 0.9
    val_and_test_frac = 0.1
    assert train_frac + val_and_test_frac == 1
    train_mask, test_mask = get_npy_one_anno_masks(input_anno, val_split=val_and_test_frac)

    # [London] our mask guarantees at least one annotation per user and item, so this is just the same as the input data now
    # item_num = train_anno.shape[0]
    # user_num = train_anno.shape[1]

    item_num = input_anno.shape[0]
    user_num = input_anno.shape[1]

    # [London] This code has been replaced by my new script (above)
    # train_idx, test_idx = (
    #     shuffle_idx[: int(0.9 * n_indices)],
    #     shuffle_idx[int(0.9 * n_indices) :],
    # )

    # train_data = {
    #     "item": (train_anno != -1).nonzero()[0][train_idx],
    #     "user": (train_anno != -1).nonzero()[1][train_idx],
    # }
    # train_label = {
    #     "rating": train_anno[
    #         (train_anno != -1).nonzero()[0][train_idx],
    #         (train_anno != -1).nonzero()[1][train_idx],
    #     ]
    # }

    train_data, train_label = boolean_mask_to_data_label(input_anno, train_mask)

    train_data = pd.DataFrame(train_data)
    train_label = pd.DataFrame(train_label)

    train_data = train_data.values.tolist()
    train_label = train_label.values.tolist()

    # [London] same for the test data
    # test_data = {
    #     "item": (train_anno != -1).nonzero()[0][test_idx],
    #     "user": (train_anno != -1).nonzero()[1][test_idx],
    # }
    # test_label = {
    #     "rating": train_anno[
    #         (train_anno != -1).nonzero()[0][test_idx],
    #         (train_anno != -1).nonzero()[1][test_idx],
    #     ]
    # }
    test_data, test_label = boolean_mask_to_data_label(input_anno, test_mask)

    test_data = pd.DataFrame(test_data)
    test_label = pd.DataFrame(test_label)

    test_data = test_data.values.tolist()
    test_label = test_label.values.tolist()

    # [London] Same for the infer data
    # infer_data = {
    #     "item": (train_anno == -1).nonzero()[0],
    #     "user": (train_anno == -1).nonzero()[1],
    # }
    # infer_label = {
    #     "rating": train_anno[
    #         (train_anno == -1).nonzero()[0], (train_anno == -1).nonzero()[1]
    #     ]
    # }
    infer_mask = get_boolean_missing_annotation_mask(input_anno)
    infer_data, infer_label = boolean_mask_to_data_label(input_anno, infer_mask)

    infer_data = pd.DataFrame(infer_data)
    infer_label = pd.DataFrame(infer_label)

    infer_data = infer_data.values.tolist()
    infer_label = infer_label.values.tolist()

    # load ratings as a dok matrix
    train_mat = np.zeros((item_num, user_num))
    temp = 0
    for x in train_data:
        train_mat[x[0], x[1]] = train_label[temp][0]
        temp += 1

    temp2 = 0
    for x in test_data:
        train_mat[x[0], x[1]] = test_label[temp2][0]
        temp2 += 1

    # load ratings as a dok matrix
    return (
        train_data,
        test_data,
        item_num,
        user_num,
        train_mat,
        train_label,
        test_label,
        infer_data,
        infer_label,
    )

class NCFData(data.Dataset):
    def __init__(
        self, features, labels, num_item, train_mat=None, num_ng=0, is_training=None
    ):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features_ps
        labels = self.labels

        item = features[idx][0]
        user = features[idx][1]
        label = labels[idx][0]
        return user, item, label