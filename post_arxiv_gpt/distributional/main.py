import openai
import os
import json
from dotenv import load_dotenv # pip install python-dotenv
from string import Formatter
import numpy as np
from tqdm import tqdm
import traceback
import time
import re

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
DATASET = "politeness"
N_SHOTS = 4
MAX_TOKENS = 300
SEED = 29
N_EXAMPLES = 30
TEMPERATURE = 0
FORMAT_TYPE = 0
ENCODING = "utf-8"
# CHANGE THIS to "no_context" if you're using the no context prompt
save_folder = f"outputs/{DATASET}_no_context_prompt/"
# CHANGE THIS to "no_context" if you're using the no context prompt
prompt_location = "./prompts/no_context_prompt.txt"
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

def format_shot(shot_index, format_type=0, imputed=False):
    shot_text = texts[shot_index][0]
    annotations_dataset = imputed_annotations if imputed else orig_annotations
    shot_distribution = single_row_label_distribution(valid_labels, annotations_dataset[shot_index])
    percent_distribution = [f"{shot_distribution[label_index] * 100:.2f}%" for label_index in range(len(valid_labels))]

    ans = f"Text: {shot_text}\nSoft labels:\n"
    for label_index in range(len(valid_labels)):
        if format_type == 0:
            ans += f"{percent_distribution[label_index]} of responsive annotators labeled the text with {valid_labels[label_index]}\n"
        else:
            raise ValueError(f"Invalid format_type {format_type}")
        
    # remove the last newline
    ans = ans[:-1]
    return ans

def format_prediction_section(example_index, format_type=0):
    target_example_text = texts[example_index][0]
    if format_type == 0:
        ans = f"Target Text: {target_example_text}\nSoft labels:"
    else:
        raise ValueError(f"Invalid format_type {format_type}")
    return ans

def response_to_distribution(response, format_type=0):
    lines = response.split("\n")
    distribution = {valid_label: 0 for valid_label in valid_labels}
    labels_covered = set()
    if format_type == 0:
        for line in lines:
            match = re.fullmatch(r"([0-9.]+)% of responsive annotators labeled the text with ([0-9]+)", line)
            if match:
                percent, label = match.groups()
                percent = float(percent)
                label = int(label)
                if label not in valid_labels:
                    raise InvalidResponseException(f"Invalid label {label} in response {response}")
                distribution[label] = percent / 100
                labels_covered.add(label)
                if len(labels_covered) == len(valid_labels):
                    # we have all the labels (sometimes it will add extra unnecessary lines)
                    break
            else:
                raise InvalidResponseException(f"Invalid line '{line}' in response:\n{response}")
            
    else:
        raise ValueError(f"Invalid format_type {format_type}")
    
    distribution = [distribution[label] for label in valid_labels]
    return distribution

# determine the target examples
np.random.seed(SEED)
selected_indices = list(np.random.choice(len(orig_annotations), N_EXAMPLES, replace=False))

model_answers = {"original": [], "imputed": [], "correct": []}
example_costs = []
for output_index, example_index in enumerate(tqdm(selected_indices)):
    correct_distribution = single_row_label_distribution(valid_labels, orig_annotations[example_index])
    model_answers["correct"].append(correct_distribution)
    for imputed_bool in [False, True]:
        imputed_bool_text = "imputed" if imputed_bool else "original"
        tqdm.write(f"Working on {imputed_bool_text} example {output_index + 1}/{N_EXAMPLES}")
        # load the dataset description
        dataset_description = file_to_string(f"./dataset_descriptions/{DATASET}_description.txt")

        # determine the target examples
        # pick N_SHOTS number of examples that are not the example_index
        np.random.seed(SEED)
        non_example_indices = np.arange(len(orig_annotations))
        non_example_indices = np.delete(non_example_indices, example_index)
        shot_indices = list(np.random.choice(non_example_indices, N_SHOTS, replace=False))

        shots_text = ""
        for shot_index in shot_indices:
            shots_text += f"Example {shot_index + 1}\n" + format_shot(shot_index, format_type=FORMAT_TYPE, imputed=imputed_bool) + "\n\n"
        shots_text = shots_text[:-2]

        prediction_section_text = format_prediction_section(example_index)

        # format the prompt
        unformatted_prompt = file_to_string(prompt_location)
        format_dict = {'dataset_description': dataset_description, 'soft_label_examples': shots_text, 'prediction_text': prediction_section_text}
        prompt = unformatted_prompt.format(**format_dict)

        messages = [
            {"role": "user", "content": prompt},
        ]

        # save the prompt
        tqdm.write("Saving the prompt...")
        save_prompt_location = os.path.join(save_folder, f"Example_{output_index}_({example_index})_{imputed_bool_text}_prompt.txt")
        make_path(save_prompt_location)
        with open(save_prompt_location, 'w', encoding=ENCODING) as f:
            json.dump(messages, f, indent=4)

        # save a clean version of the prompt
        save_clean_prompt_location = os.path.join(save_folder, f"Example_{output_index}_({example_index})_{imputed_bool_text}_clean_prompt.txt")
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
        response_save_location = os.path.join(save_folder, f"Example_{output_index}_({example_index})_{imputed_bool_text}_response.json")
        make_path(response_save_location)
        with open(response_save_location, "w", encoding=ENCODING) as f:
            json.dump(response, f, indent=4)
        tqdm.write("Response saved.")

        response_text = response.choices[0].message.content
        model_answer = response_to_distribution(response_text, format_type=FORMAT_TYPE)
        model_answers[imputed_bool_text].append(model_answer)

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

assert len(model_answers["original"]) == len(model_answers["imputed"]) == len(model_answers["correct"])

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

# compute the KL divergence
kl_divergences = {"raw": {"original": [], "imputed": []}}
for i in range(len(model_answers["correct"])):
    for imputed_bool in [False, True]:
        imputed_bool_text = "imputed" if imputed_bool else "original"
        kl_divergence_score = kl_divergence(np.array(model_answers["correct"][i]), np.array(model_answers[imputed_bool_text][i]))
        kl_divergences["raw"][imputed_bool_text].append(kl_divergence_score)

# get the mean and standard deviation of the KL divergences
kl_divergences["means"] = {"original": np.mean(kl_divergences["raw"]["original"]), "imputed": np.mean(kl_divergences["raw"]["imputed"])}
kl_divergences["stds"] = {"original": np.std(kl_divergences["raw"]["original"]), "imputed": np.std(kl_divergences["raw"]["imputed"])}

with open(make_path(os.path.join(save_folder, "statistics.json")), "w", encoding=ENCODING) as f:
    json.dump(kl_divergences, f, indent=4)

# clean kl_divergence statistics
clean_kl_divergences = {imputed_bool_text: f"{kl_divergences['means'][imputed_bool_text]:.3f} ± {kl_divergences['stds'][imputed_bool_text]:.3f}" for imputed_bool_text in ["original", "imputed"]}

# save clean kl_divergence statistics
with open(make_path(os.path.join(save_folder, "clean_statistics.json")), "w", encoding=ENCODING) as f:
    json.dump(clean_kl_divergences, f, indent=4)

print("Testing complete.")
print("Statistics:")
print(clean_kl_divergences)
print("Total Price")
print(example_costs["total_cost"])