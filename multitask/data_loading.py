import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer
import warnings

# import from kernel_matrix_factorization
import sys
sys.path.append('../kernel_matrix_factorization')
from dataset_formats import NpyAnnotationsFormat, NpyTextsFormat

class RatingDataset(TensorDataset):
    def __init__(self, input_ids, attention_mask, token_type_ids, annotations):
        super().__init__(input_ids, attention_mask, token_type_ids, annotations)
        self.annotations = annotations

        # warn the user if their dataset has a row of all missing annotations
        if torch.all(self.annotations == -1, axis=1).any():
            # warn the user if all their annotations are missing
            if (annotations == -1).all():
                warnings.warn("You created a dataset with all missing annotations. This usually occurs because you decided to train on all of the available data (usually during imputation), so the validation dataset is empty. If that's not the case, then you should check your dataset to make sure that it is correct.")
            else:
                warnings.warn("You created a dataset with a row of all missing annotations.")

def tokenize_texts(texts, tokenizer):
    inputs = tokenizer(texts, return_tensors="pt", padding='max_length', truncation=True)
    return inputs

def load_data_from_paths(train_annotations_path, train_texts_path, val_annotations_path, val_texts_path, encoder_model: str):
    print("Loading all datasets")
    train_annotations = np.load(train_annotations_path, allow_pickle=True)
    train_texts = np.load(train_texts_path, allow_pickle=True).reshape((-1,)).tolist()
    val_annotations = np.load(val_annotations_path, allow_pickle=True)
    val_texts = np.load(val_texts_path, allow_pickle=True).reshape((-1,)).tolist()
    print("Datasets loaded")
    return load_data(train_annotations, train_texts, val_annotations, val_texts, encoder_model)

def load_data(train_annotations, train_texts, val_annotations, val_texts, encoder_model: str):
    tokenizer = get_tokenizer(encoder_model)
    print("Loading train...")
    train_loader = load_one_data(train_annotations, train_texts, tokenizer, num_workers=4)
    print("Train loading complete.")
    print("Loading val...")
    val_loader = load_one_data(val_annotations, val_texts, tokenizer, num_workers=0)
    print("Val loading complete.")
    return train_loader, val_loader

def get_tokenizer(encoder_model: str):
    return AutoTokenizer.from_pretrained(encoder_model, use_fast=False)

def load_one_data(annotations: NpyAnnotationsFormat, texts: NpyTextsFormat, tokenizer, num_workers=4):
    assert annotations.shape[0] == len(texts), f"Mismatch in the number of annotations and texts: {annotations.shape[0]} vs {len(texts)}."
    annotations = np.round(annotations).astype(int)
    texts = np.array(texts).reshape((-1,)).tolist()

    print("Tokenizing inputs...")
    inputs = tokenize_texts(texts, tokenizer)
    print("Tokenization complete")

    dataset = RatingDataset(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'], torch.tensor(annotations, dtype=torch.long))

    # Use num_workers=0 for test data to help with reproducibility
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers)

    return loader