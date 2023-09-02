import openai
import os
import json
from dotenv import load_dotenv # pip install python-dotenv
from string import Formatter
import numpy as np
from tqdm import tqdm
import traceback
import time
from typing import Union
from enum import Enum

# Load the secret file
load_dotenv("../secrets/openai.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "You did not put the OPENAI_API_KEY in the openai.env file. The openai.env file should look like 'OPENAI_API_KEY=sk-...'"
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
assert OPENAI_ORGANIZATION, "You did not put the OPENAI_ORGANIZATION in the openai.env file. The openai.env file should look like 'OPENAI_ORGANIZATION=org-...'"
openai.api_key = OPENAI_API_KEY
openai.organization = OPENAI_ORGANIZATION

GPT4_ID = "gpt-4" # chat only
CHAT_GPT_ID = "gpt-3.5-turbo"
GPT3_ID = "text-davinci-003"

PROMPT_PRICE_PER_TOKEN_KEY = "prompt_price_per_token"
COMPLETION_PRICE_PER_TOKEN_KEY = "completion_price_per_token"
GENERAL_PRICE_PER_TOKEN_KEY = "price_per_token"

pricing = {GPT4_ID: {PROMPT_PRICE_PER_TOKEN_KEY: 0.03 / 1000, COMPLETION_PRICE_PER_TOKEN_KEY: 0.06 / 1000}, GPT3_ID: {GENERAL_PRICE_PER_TOKEN_KEY: 0.02 / 1000}}

# How many requests per minute can we make to each model?
# https://platform.openai.com/docs/guides/rate-limits/overview
rate_limits = {GPT3_ID: 20}

def file_to_string(filename: str) -> str:
    with open(filename, "r", encoding='utf-8') as f:
        return f.read()
    
def get_format_names(s):
    # from https://stackoverflow.com/questions/22830226/how-to-get-the-variable-names-from-the-string-for-the-format-method
    names = [fn for _, fn, _, _ in Formatter().parse(s) if fn is not None]

    return names

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Set these values (and one thing below too)
MODEL=GPT3_ID
DATASET="Fixed2xSBIC"
# folder for the original dataset with '{dataset}' and '{annotations_or_texts}' as placeholders
DATASET_LOCATION="../datasets/cleaned/{dataset}_{annotations_or_texts}.npy"
# imputed dataset location with '{dataset}' as a placeholder
IMPUTED_DATASET_LOCATION="../datasets/final_imputation_results/ncf_imputation_results/{dataset}_-1_ncf_imputation_-1.npy"
# how many shots from the invididual
N_SHOTS: int=4
# How many annotators to use. This is how many outputs there will be.
N_ANNOTATORS: int=30
MAX_TOKENS=5
SEED=47

class ExtraDataType(Enum):
    '''What type of extra data (in addition to how the user responds to other questions)
    should be added to the prompt
    If k_shots is a number, then we just add that many shots from other users.
    Otherwise, if it's one of these values, we use this.
    '''
    # add the majority voted label
    MAJORITY_VOTED = "majority_voted"
    # give the model the original distributional/soft label
    ORIGINAL_DISTRIBUTIONAL = "original_distributional_correct"
    # give the model the imputed distributional/soft label
    IMPUTED_DISTRIBUTIONAL = "imputed_distributional_correct"
    # don't just do k shots, do all shots
    ALL_SHOTS = "all_shots"

def make_str_k_shots(k_shots: Union[int, ExtraDataType]) -> str:
    if isinstance(k_shots, int):
        return str(k_shots)
    else:
        return k_shots.value

# What type of data should the shots include?
# YOU SHOULD ALSO SET THIS
k_shots: Union[int, ExtraDataType]=ExtraDataType.ORIGINAL_DISTRIBUTIONAL
str_k_shots = make_str_k_shots(k_shots)

save_folder=f"./output/{DATASET}_Seed_{SEED}_K_{str_k_shots}/"
if k_shots == ExtraDataType.MAJORITY_VOTED:
    prompt_file = "prompts/majority_voted.txt"
elif isinstance(k_shots, int) or k_shots == ExtraDataType.ALL_SHOTS:
    prompt_file = "prompts/individual.txt"
elif k_shots in (ExtraDataType.ORIGINAL_DISTRIBUTIONAL, ExtraDataType.IMPUTED_DISTRIBUTIONAL):
    prompt_file = "prompts/distributional.txt"

make_path(save_folder)

annotations = np.load(DATASET_LOCATION.format(dataset=DATASET, annotations_or_texts='annotations'), allow_pickle=True)
if k_shots == ExtraDataType.IMPUTED_DISTRIBUTIONAL:
    imputed_annotations = np.load(IMPUTED_DATASET_LOCATION.format(dataset=DATASET), allow_pickle=True)
texts = np.load(DATASET_LOCATION.format(dataset=DATASET, annotations_or_texts='texts'), allow_pickle=True)

# remove any columns with less than N_SHOTS + 1 examples
# Create boolean mask for condition
mask = np.sum(annotations != -1, axis=0) >= N_SHOTS + 1

# Get columns that satisfy the condition
selected_cols = annotations[:, mask]

# Get indices of selected columns in the original array
original_cols = np.where(mask)[0]

# assert that there are at least N_ANNOTATORS columns
assert len(original_cols) >= N_ANNOTATORS, f"There are only {len(original_cols)} columns with at least {N_SHOTS + 1} examples. Please decrease N_SHOTS or decrease N_ANNOTATORS."

# Select the annotators we're going to use
np.random.seed(SEED)
selected_cols = np.random.choice(original_cols, N_ANNOTATORS, replace=False)
np.save(os.path.join(save_folder, "selected_indices.npy"), selected_cols, allow_pickle=True)

all_examples = []
test_example_lines = []
test_answers = []
test_answer_rows = []
for annotator_col in selected_cols:
    shots = ""
    for shot_index in range(N_SHOTS + 1):
        # UPGRADE ideally the order of the examples should be random
        # get the row of the shot_index-th example for the annotator_index-th annotator
        # this will be the shot_index-th non-empty row in the annotator_index-th column
        row_index = np.where(annotations[:, annotator_col] != -1)[0][shot_index]
        shot_answer = int(annotations[row_index, annotator_col])
        shot_text = texts[row_index][0]

        if shot_index == N_SHOTS:
            # this is the test example
            test_answers.append(shot_answer)
            test_answer_rows.append(row_index)
            test_example_lines.append(f"EXAMPLE: {shot_text}")
        else:
            shots += f"\n{shot_index + 1}.\nEXAMPLE: {shot_text}\nANSWER: {shot_answer}\n"
    # remove the final and initial newline
    shots = shots[1:-1]
    all_examples.append(shots)

assert len(all_examples) == N_ANNOTATORS, f"{len(all_examples)} != {N_ANNOTATORS}"
assert len(test_answers) == N_ANNOTATORS, f"{len(test_answers)} != {N_ANNOTATORS}"
assert len(test_answer_rows) == N_ANNOTATORS, f"{len(test_answer_rows)} != {N_ANNOTATORS}"

# get annotations for the test example from the other annotators
other_examples = []
ks = []
for annotator_index, annotator_col in enumerate(selected_cols):
    row_index = test_answer_rows[annotator_index]

    if k_shots == ExtraDataType.MAJORITY_VOTED:
        other_annotator_indices = np.where(annotations[row_index] != -1)[0]
        # will return the first value seen if there's a tie
        maj_vote = np.bincount(annotations[row_index, other_annotator_indices].astype(int)).argmax()
        other_examples.append(f"1.\nEXAMPLE: {texts[row_index][0]}\nANSWER: {maj_vote}")
        ks.append(1)
    elif k_shots == ExtraDataType.ORIGINAL_DISTRIBUTIONAL or ExtraDataType.IMPUTED_DISTRIBUTIONAL:
        if k_shots == ExtraDataType.ORIGINAL_DISTRIBUTIONAL:
            ks_add = 'original_distributional_label'
            dataset = annotations
        else:
            ks_add = 'imputed_distributional_label'
            dataset = imputed_annotations

        # get the distributional label (minus this annotator) for the test example
        other_annotator_indices = np.where((annotations[row_index] != -1) & (np.arange(annotations.shape[1]) != annotator_col))[0]
        dist_label = np.bincount(dataset[row_index, other_annotator_indices].astype(int))
        # convert to percentages
        dist_label = dist_label / np.sum(dist_label)
        shots = ""
        for label, percentage in enumerate(dist_label):
            percentage *= 100
            shots += f"\n{percentage:.2f}% of people responded with {label}"
        
        # remove the initial newline
        shots = shots[1:]
        other_examples.append(shots)
        ks.append(ks_add)

    elif isinstance(k_shots, int) or k_shots == ExtraDataType.ALL_SHOTS:
        shots = ""

        # get K_SHOTS columns that aren't the annotator_index-th column and aren't -1s (missing)
        other_annotator_indices = np.where((annotations[row_index] != -1) & (np.arange(annotations.shape[1]) != annotator_col))[0]
        try:
            k = k_shots if k_shots is not ExtraDataType.ALL_SHOTS else len(other_annotator_indices)
            other_annotator_indices = np.random.choice(other_annotator_indices, k, replace=False)
            ks.append(k)
        except ValueError:
            # there aren't enough other annotators
            raise ValueError(f"There aren't enough other annotators to get {k} examples for the test example {row_index}. There are only {len(other_annotator_indices)} other annotators.")

        for shot_index, other_annotator_index in enumerate(other_annotator_indices):
            shot_answer = int(annotations[row_index, other_annotator_index])
            shot_text = texts[row_index][0]

            shots += f"\n{shot_index + 1}.\nEXAMPLE: {shot_text}\n"
            shots += f"ANSWER: {shot_answer}\n"

        # remove the final and initial newline
        shots = shots[1:-1]
        other_examples.append(shots)
    else:
        raise ValueError(f"Invalid k_shots value: {k_shots}")

assert len(other_examples) == N_ANNOTATORS, f"{len(other_examples)} != {N_ANNOTATORS}"

model_answers = []
annotator_costs = []
# how many seconds to pause so that we don't hit the rate limit
rate_limit_pause = 60 / rate_limits[MODEL]
for annotator_col in tqdm(range(N_ANNOTATORS)):
    prompt = file_to_string(prompt_file)
    format_values = {}
    for format_name in get_format_names(prompt):
        if format_name.lower() == "shots":
            format_values[format_name] = all_examples[annotator_col]
        elif format_name.lower() == "other_shots":
            format_values[format_name] = other_examples[annotator_col]
        elif format_name.lower() == "target_example_line":
            format_values[format_name] = test_example_lines[annotator_col]
        elif format_name.lower() == "n_shots":
            format_values[format_name] = N_SHOTS
        elif format_name.lower() == "k_shots":
            # UPGRADE I don't think this is ever used
            format_values[format_name] = ks[annotator_col]
        elif format_name.lower() == "dataset_description":
            format_values[format_name] = file_to_string(f"dataset_descriptions/{DATASET}_description.txt")
        else:
            format_values[format_name] = file_to_string(format_name)

    prompt = prompt.format(**format_values)

    with open(os.path.join(save_folder, f"Annotator_{annotator_col}_({selected_cols[annotator_col]})_prompt.txt"), "w", encoding='utf-8') as f:
        f.write(prompt)

    # Now, have the model complete the prompt
    tqdm.write("Asking OpenAI to complete the prompt...")
    response = openai.Completion.create(
        model=MODEL,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        temperature=0,
        logprobs=5
    )
    tqdm.write("OpenAI has completed the prompt.")
    tqdm.write("Saving the response...")
    # save the json response
    with open(os.path.join(save_folder, f"Annotator_{annotator_col}_({selected_cols[annotator_col]})_response.json"), "w", encoding='utf-8') as f:
        json.dump(response, f)
    tqdm.write("Response saved.")

    model_answer = response.choices[0].text
    try:
        model_answer = int(model_answer.strip())
    except:
        model_answer = None

    model_answers.append(model_answer)

    try:
        price_per_token = pricing[MODEL][GENERAL_PRICE_PER_TOKEN_KEY]
        annotator_cost = price_per_token * response.usage.total_tokens
        annotator_costs.append(annotator_cost)
    except Exception as e:
        tb_str = traceback.format_exc()
        tqdm.write("Something went wrong when trying to compute the cost.")
        tqdm.write(tb_str)

    tqdm.write(f"Total cost so far: {np.sum(annotator_costs)}")
    # pause so that we don't hit the rate limit
    tqdm.write(f"Pausing for {rate_limit_pause} seconds so that we don't hit the rate limit...")
    time.sleep(rate_limit_pause)

# save the annotator costs as json
annotator_costs_json = {
    "annotator_costs": annotator_costs,
    "total_cost": np.sum(annotator_costs)
}
with open(os.path.join(save_folder, "annotator_costs.json"), "w", encoding='utf-8') as f:
    json.dump(annotator_costs_json, f)

# create an array with two rows
# the first row is the model answers
# the second row is the test answers
answers = np.array([model_answers, test_answers])

# save the answers in a numpy file
np.save(os.path.join(save_folder, "answers.npy"), answers, allow_pickle=True)

# save the answers in a json file
answers_json = {
    "model_answers": model_answers,
    "test_answers": test_answers
}
with open(os.path.join(save_folder, "answers.json"), "w", encoding='utf-8') as f:
    json.dump(answers_json, f)

# compute the accuracy
accuracy = np.sum(answers[0] == answers[1]) / N_ANNOTATORS

# compute the rmse
rmse = np.sqrt(np.sum((answers[0] - answers[1]) ** 2) / N_ANNOTATORS)

# save the statistics in a json file
statistics = {
    "accuracy": accuracy,
    "rmse": rmse,
    "n": N_ANNOTATORS
}

with open(os.path.join(save_folder, "statistics.json"), "w", encoding='utf-8') as f:
    json.dump(statistics, f)

print("Testing complete.")
print("Statistics:")
print(statistics)
print("Price")
print(annotator_costs_json)