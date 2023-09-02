import numpy as np
import pandas as pd
from dataset_formats import NpyAnnotationsFormat, NpyTextsFormat
from typing import Tuple
from tqdm import tqdm
from make_path import make_path

def fix_duplicate_texts(annotations: NpyAnnotationsFormat, texts: NpyTextsFormat, randomize=False) -> Tuple[NpyAnnotationsFormat, NpyTextsFormat]:
    '''
    Sometimes the npy files have multiple rows that correspond to the same example.
    This function combines the rows that correspond to the same example, and errors out if there are any conflicting annotations/labels, unless randomize is True, in which case it picks an answer at random.
    '''
    # Combine annotations and texts into a single DataFrame
    combined_dataframe = pd.concat(
        [pd.DataFrame(annotations), pd.DataFrame(texts, columns=["sentence"])],
        axis=1,
    )

    # Identify the unique sentences
    unique_sentences = combined_dataframe["sentence"].unique()

    # Collect the rows with unique texts and no conflicting annotations
    cleaned_rows = []
    for sentence in tqdm(unique_sentences):
        sentence_rows = combined_dataframe[combined_dataframe["sentence"] == sentence]

        for col_index in range(len(annotations[0])):
            unique_values = sentence_rows.iloc[:, col_index].unique()
            unique_values = unique_values[unique_values != -1]  # Ignore missing values

            if len(unique_values) > 1:
                if randomize:
                    # Randomly select an annotation to use
                    tqdm.write(
                        f"Conflicting annotations found for sentence '{sentence}' and column {col_index}: {unique_values}. Randomly selecting one."
                    )
                    sentence_rows.iloc[0, col_index] = np.random.choice(unique_values)
                else:
                    raise ValueError(
                        f"Conflicting annotations found for sentence '{sentence}' and column {col_index}: {unique_values}."
                    )

        cleaned_rows.append(sentence_rows.iloc[0])

    # Create cleaned annotations and texts npy files
    cleaned_dataframe = pd.concat(cleaned_rows, axis=1).transpose()
    cleaned_annotations = cleaned_dataframe.drop(columns=["sentence"]).to_numpy()
    cleaned_texts = cleaned_dataframe["sentence"].to_numpy().reshape(-1, 1)

    # Set the correct data types
    cleaned_annotations = cleaned_annotations.astype(annotations.dtype)
    cleaned_texts = cleaned_texts.astype(texts.dtype)

    return cleaned_annotations, cleaned_texts

    
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to fix the duplicate texts for",
    )
    parser.add_argument(
        "--load_location",
        type=str,
        default="../datasets",
        required=False,
        help="Folder where the npy files are located. The files should be named <dataset_name>_annotations.npy and <dataset_name>_texts.npy",
    )
    parser.add_argument(
        "--save_location",
        type=str,
        default="../datasets/cleaned",
        required=False,
        help="Folder where the cleaned npy files should be saved. The files will be named <dataset_name>_annotations_cleaned.npy and <dataset_name>_texts_cleaned.npy",
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="If set, will randomly select an annotation to use when there are conflicting annotations for a sentence, rather than erroring out.",
    )
    args = parser.parse_args()

    annotations_npy_file = os.path.join(args.load_location, f"{args.dataset_name}_annotations.npy")
    texts_npy_file = os.path.join(args.load_location, f"{args.dataset_name}_texts.npy")

    with open(annotations_npy_file, "rb") as f:
        annotations = np.load(f, allow_pickle=True)

    with open(texts_npy_file, "rb") as f:
        texts = np.load(f, allow_pickle=True)

    cleaned_annotations, cleaned_texts = fix_duplicate_texts(annotations, texts, randomize=args.randomize)

    # save the cleaned annotations and texts npy files
    cleaned_annotations_npy_file = os.path.join(args.save_location, f"{args.dataset_name}_annotations_cleaned.npy")
    cleaned_texts_npy_file = os.path.join(args.save_location, f"{args.dataset_name}_texts_cleaned.npy")

    make_path(cleaned_annotations_npy_file)
    make_path(cleaned_texts_npy_file)

    with open(cleaned_annotations_npy_file, "wb") as f:
        np.save(f, cleaned_annotations)

    with open(cleaned_texts_npy_file, "wb") as f:
        np.save(f, cleaned_texts)

    print(f"Cleaned annotations npy file: {cleaned_annotations_npy_file}")
    print(f"Cleaned texts npy file: {cleaned_texts_npy_file}")