'''Annotator tagging version'''

import openai
import os
import json
from dotenv import load_dotenv # pip install python-dotenv
from string import Formatter
import numpy as np
from tqdm import tqdm
import traceback
import time
from sklearn.metrics import classification_report

# import from disagreement_shift
import sys
sys.path.append("../../disagreement_shift/label_distribution")
from label_distribution import single_row_label_distribution

# import from multitask
sys.path.append("../../multitask")
from numpy_json_encoder import NumpyJSONEncoder

# Load the secret file
load_dotenv("../../secrets/openai.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "You did not put the OPENAI_API_KEY in the secret.env file. The secret.env file should look like 'OPENAI_API_KEY=sk-...'"
OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
assert OPENAI_ORGANIZATION, "You did not put the OPENAI_ORGANIZATION in the secret.env file. The secret.env file should look like 'OPENAI_ORGANIZATION=org-...'"
openai.api_key = OPENAI_API_KEY
openai.organization = OPENAI_ORGANIZATION

GPT4_ID = "gpt-4" # chat only
CHAT_GPT_ID = "gpt-3.5-turbo"
GPT3_ID = "text-davinci-003"

PROMPT_PRICE_PER_TOKEN_KEY = "prompt_price_per_token"
COMPLETION_PRICE_PER_TOKEN_KEY = "completion_price_per_token"
GENERAL_PRICE_PER_TOKEN_KEY = "price_per_token"

pricing = {GPT4_ID: {PROMPT_PRICE_PER_TOKEN_KEY: 0.03 / 1000, COMPLETION_PRICE_PER_TOKEN_KEY: 0.06 / 1000}, GPT3_ID: {GENERAL_PRICE_PER_TOKEN_KEY: 0.02 / 1000}, CHAT_GPT_ID: {PROMPT_PRICE_PER_TOKEN_KEY: 0.002 / 1000, COMPLETION_PRICE_PER_TOKEN_KEY: 0.002 / 1000}}

# How many requests per minute can we make to each model?
# https://platform.openai.com/docs/guides/rate-limits/overview
rate_limits = {GPT3_ID: 20, GPT4_ID: 200, CHAT_GPT_ID: 200}

def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon=1e-10):
    '''Takes two 1D or 2D arrays and returns the KL divergence between them'''
    if len(p.shape) == 1:
        axis = None
    elif len(p.shape) == 2:
        axis = 1
    else:
        raise ValueError(f"Invalid shape {p.shape}. Must have 1 or 2 dimensions, had {len(p.shape)} dimensions instead")
    ans = np.sum(p * np.log((p + epsilon) / (q + epsilon)), axis=axis)
    return ans

def file_to_string(filename: str) -> str:
    with open(filename, "r", encoding='utf-8') as f:
        return f.read()
    
def get_format_names(s):
    # from https://stackoverflow.com/questions/22830226/how-to-get-the-variable-names-from-the-string-for-the-format-method
    names = [fn for _, fn, _, _ in Formatter().parse(s) if fn is not None]

    return names

def make_path(filename):
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    return filename

class InvalidResponseException(RuntimeError):
    pass

# Set these values
MODEL = CHAT_GPT_ID
DATASET = "Fixed2xSBIC"
N_SHOTS = 4
MAX_TOKENS = 300
SEED = 29
N_EXAMPLES = 30
TEMPERATURE = 0
FORMAT_TYPE = 0
ENCODING = "utf-8"
ANNOTATOR_LABELS = ["A", "B", "C"]
prompt_name = "annotator_tagging"
assert prompt_name == "annotator_tagging", "This code is only meant to be used with the annotator_tagging prompt. If you've written an extremely similar prompt and want to try it out, you can delete this assert (or rename your prompt)."
save_folder = f"outputs/{DATASET}_{prompt_name}/"
prompt_location = f"./prompts/{prompt_name}.txt"
orig_annotations_location = f"../../datasets/cleaned/{DATASET}_annotations.npy"
imputed_annotations_location = f"../../datasets/final_imputation_results/ncf_imputation_results/{DATASET}_-1_ncf_imputation_-1.npy"
texts_location = f"../../datasets/cleaned/{DATASET}_texts.npy"
rate_limit_pause = 60 / rate_limits[MODEL]

# Load the data
orig_annotations = np.load(orig_annotations_location, allow_pickle=True)
imputed_annotations = np.load(imputed_annotations_location, allow_pickle=True)
texts = np.load(texts_location, allow_pickle=True)
# will round the labels to the nearest integer
valid_labels = sorted(list(map(int, np.unique(orig_annotations[orig_annotations != -1]))))

# pick N_EXAMPLES random examples that have at least 3 annotators
valid_example_idxs = np.asarray(np.sum(orig_annotations != -1, axis=1) >= N_SHOTS).nonzero()[0]
np.random.seed(SEED)
example_idxs = np.random.choice(valid_example_idxs, size=N_EXAMPLES, replace=False)

def format_examples(annotator_id, example_ids, imputed=False):
    ans = ""
    for example_count, example_idx in enumerate(example_ids):
        ans += f"Example {example_count + 1}:\n"
        ans += f"Text: {texts[example_idx][0]}\n"
        load_dataset = imputed_annotations if imputed else orig_annotations
        annotation = load_dataset[example_idx, annotator_id]
        assert annotation != -1, f"Annotation for annotator {annotator_id} on example {example_idx} was -1"
        ans += f"Annotation from annotator: {annotation}\n\n"

    ans = ans[:-2] # remove the last newlines
    return ans

def format_target_example(example_idx):
    ans = f"Text: {texts[example_idx][0]}\nAnnotation from annotator:"
    return ans
        
dataset_description = file_to_string(f"../distributional/dataset_descriptions/{DATASET}_description.txt")        

model_answers = {"original": {"A": [], "B": [], "C": []}, "imputed": {"A": [], "B": [], "C": []}, "correct": {"A": [], "B": [], "C": []}}
example_costs = []
for example_count, example_idx in enumerate(tqdm(example_idxs)):
    # determine 3 annotators at random
    valid_annotator_idxs = np.asarray(orig_annotations[example_idx] != -1).nonzero()[0]
    # whittle down to only annotators that have labeled at least N_SHOTS + 1 examples
    valid_annotator_idxs = valid_annotator_idxs[np.asarray(np.sum(orig_annotations[:, valid_annotator_idxs] != -1, axis=0) >= N_SHOTS + 1).nonzero()[0]]
    np.random.seed(SEED)
    annotator_idxs = np.random.choice(valid_annotator_idxs, size=3, replace=False)

    model_answers["correct"]["A"].append(orig_annotations[example_idx, annotator_idxs[0]])
    model_answers["correct"]["B"].append(orig_annotations[example_idx, annotator_idxs[1]])
    model_answers["correct"]["C"].append(orig_annotations[example_idx, annotator_idxs[2]])

    for imputed_bool in [False, True]:
        imputed_bool_text = "imputed" if imputed_bool else "original"
        load_dataset = imputed_annotations if imputed_bool else orig_annotations

        # grab N_SHOTS examples that are not the target example for each annotator
        annotator_examples = []
        for annotator_idx in annotator_idxs:
            annotator_example_idxs = np.asarray(load_dataset[:, annotator_idx] != -1).nonzero()[0]
            annotator_example_idxs = annotator_example_idxs[annotator_example_idxs != example_idx]
            np.random.seed(SEED)
            annotator_examples.append(np.random.choice(annotator_example_idxs, size=N_SHOTS, replace=False))
            assert load_dataset[annotator_examples[-1][-1], annotator_idx] != -1, f"Annotator {annotator_idx} has an example that was not labeled"
            assert load_dataset[annotator_examples[-1][0], annotator_idx] != -1, f"Annotator {annotator_idx} has an example that was not labeled"

        assert len(annotator_examples) == 3
            
        annotator_A_examples = format_examples(annotator_idxs[0], annotator_examples[0], imputed=imputed_bool)
        annotator_B_examples = format_examples(annotator_idxs[1], annotator_examples[1], imputed=imputed_bool)
        annotator_C_examples = format_examples(annotator_idxs[2], annotator_examples[2], imputed=imputed_bool)

        target_example = format_target_example(example_idx)
        for target_annotator_label in ANNOTATOR_LABELS:
            format_dict = {"dataset_description": dataset_description, "annotator_A_examples": annotator_A_examples, "annotator_B_examples": annotator_B_examples, "annotator_C_examples": annotator_C_examples, "target_annotator": target_annotator_label, "target_example": target_example}

            unformatted_prompt = file_to_string(prompt_location)
            prompt = unformatted_prompt.format(**format_dict)

            messages = [
                {"role": "user", "content": prompt},
            ]

            # save the prompt
            tqdm.write("Saving the prompt...")
            save_prompt_location = os.path.join(save_folder, f"Example_{example_count}_({example_idx})_{target_annotator_label}_{imputed_bool_text}_prompt.txt")
            make_path(save_prompt_location)
            with open(save_prompt_location, 'w', encoding=ENCODING) as f:
                json.dump(messages, f, indent=4)

            # save a clean version of the prompt
            save_clean_prompt_location = os.path.join(save_folder, f"Example_{example_count}_({example_idx})_{target_annotator_label}_{imputed_bool_text}_clean_prompt.txt")
            make_path(save_clean_prompt_location)
            with open(save_clean_prompt_location, 'w', encoding=ENCODING) as f:
                f.write(prompt)

            # Now, have the model complete the prompt
            tqdm.write("Asking OpenAI to complete the prompt...")
            response = openai.ChatCompletion.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
                temperature=TEMPERATURE
            )

            tqdm.write("OpenAI has completed the prompt.")
            # save the json response
            tqdm.write("Saving the response...")
            response_save_location = os.path.join(save_folder, f"Example_{example_count}_({example_idx})_{target_annotator_label}_{imputed_bool_text}_response.json")
            make_path(response_save_location)
            with open(response_save_location, "w", encoding=ENCODING) as f:
                json.dump(response, f, indent=4)

            response_text = response.choices[0].message.content

            # save clean version of the response
            clean_response_save_location = os.path.join(save_folder, f"Example_{example_count}_({example_idx})_{target_annotator_label}_{imputed_bool_text}_clean_response.txt")
            with open(make_path(clean_response_save_location), "w", encoding=ENCODING) as f:
                f.write(response_text)
            tqdm.write("Response saved.")

            try:
                model_answer = float(response_text.strip())
            except:
                model_answer = None

            model_answers[imputed_bool_text][target_annotator_label].append(model_answer)

            try:
                prompt_price_per_token = pricing[MODEL][PROMPT_PRICE_PER_TOKEN_KEY]
                completion_price_per_token = pricing[MODEL][COMPLETION_PRICE_PER_TOKEN_KEY]
                cost = response.usage.prompt_tokens * prompt_price_per_token + response.usage.completion_tokens * completion_price_per_token
            except Exception as e:
                tb_str = traceback.format_exc()
                tqdm.write("Something went wrong when trying to compute the cost.")
                tqdm.write(tb_str)

            tqdm.write(f"Cost: {cost}")
            example_costs.append(cost)

            # pause so that we don't hit the rate limit
            tqdm.write(f"Pausing for {rate_limit_pause} seconds so that we don't hit the rate limit...")
            time.sleep(rate_limit_pause)

# save the annotator costs as json
example_costs = {
    "example_costs": example_costs,
    "total_cost": float(np.sum(example_costs))
}
with open(make_path(os.path.join(save_folder, "example_costs.json")), "w", encoding=ENCODING) as f:
    json.dump(example_costs, f, indent=4)
    
# save the answers in a json file
with open(make_path(os.path.join(save_folder, "answers.json")), "w", encoding=ENCODING) as f:
    json.dump(model_answers, f, indent=4, cls=NumpyJSONEncoder)

# compute classification report across imputed and original

original_predictions = [pred if pred is not None else -1 for annotator_label in ANNOTATOR_LABELS for pred in model_answers["original"][annotator_label]]
imputed_predictions = [pred if pred is not None else -1 for annotator_label in ANNOTATOR_LABELS for pred in model_answers["imputed"][annotator_label]]
correct_labels = [pred for annotator_label in ANNOTATOR_LABELS for pred in model_answers["correct"][annotator_label]]

original_classification_report = classification_report(correct_labels, original_predictions, output_dict=True, zero_division=0)
imputed_classification_report = classification_report(correct_labels, imputed_predictions, output_dict=True, zero_division=0)

all_data = {
    "original": original_classification_report,
    "imputed": imputed_classification_report
}

with open(make_path(os.path.join(save_folder, "statistics.json")), "w", encoding=ENCODING) as f:
    json.dump(all_data, f, indent=4)

# summarize the main stats
# same as above but with comprehensions
clean_stats = {
    imputed_bool_text: {
        "accuracy": all_data[imputed_bool_text]["accuracy"],
        "weighted_f1": all_data[imputed_bool_text]["weighted avg"]["f1-score"],
    }
    for imputed_bool_text in all_data
}

with open(make_path(os.path.join(save_folder, "clean_statistics.json")), "w", encoding=ENCODING) as f:
    json.dump(clean_stats, f, indent=4)

print("Testing complete.")
print("Statistics:")
print(clean_stats)
print("Total Price")
print(example_costs["total_cost"])