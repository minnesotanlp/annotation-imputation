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
# ran into issues with ChatGPT and the token rate limit, so making it smaller
# making ChatGPT even smaller so that I can (hopefully) do two experiments at once
rate_limits = {GPT3_ID: 20, GPT4_ID: 200, CHAT_GPT_ID: 45}

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
# DATASETS = ["politeness", "Fixed2xSBIC"]
DATASETS = ["ghc", "SChem", "Sentiment", "SChem5Labels", "politeness", "Fixed2xSBIC"]
# DATASETS = ["ghc"] # just for testing
MAX_TOKENS = 2
SEED = 29
N_IMPUTED_EXAMPLES = 30
N_ORIG_EXAMPLES = 30
N_ANNOTATORS = 30
# N_ANNOTATORS = 3 # just for testing
TEMPERATURE = 0
ENCODING = "utf-8"

# set the seed 
np.random.seed(SEED)

# PROMPT_NAMES = ["combined1", "just_orig1", "just_imputed1", "combined3", "just_imputed3", "just_orig3"]
PROMPT_NAMES = ["just_orig7", "just_orig6", "just_orig5", "combined3", "just_imputed1", "just_imputed3"]

# make sure that all prompts exist so it doesn't crash later
for prompt_name in PROMPT_NAMES:
    assert os.path.exists(f"./prompts/{prompt_name}.txt"), f"Prompt {prompt_name} does not exist (could not be found at {os.path.abspath(f'./prompts/{prompt_name}.txt')})"

imputed_examples_headers = [
    "Estimated Examples:",
    "Imputed Examples:"
]

orig_examples_headers = [
    "Correct Examples:",
    "Human-rated Examples:",
    "Human-Rated Examples:",
    "Human-rated Examples: (there may not be any)",
    "Examples from the dataset (there may not be any):"
    "Examples from the dataset (there may not be any)",
]

target_examples_headers = [
    "Target Example:"
]

instructions = [
    "Predict the integer label of the following example:",
    "Now you will make your prediction (if you are unsure, just give your best estimate.) Your output should be an integer label:"
]

final_words = [
    "Your output should be a single integer corresponding to the label.",
    "Your output should be a single integer and nothing else.",
    "The only valid output is a single integer.",
    "If you output anything other than a single integer, your output will be considered invalid."
    "If you output anything other than a single integer, your output will harm the integrity of our dataset.",
    "If you output anything other than a single integer (and absolutely nothing else, including explanatory text), your output will invalidate the dataset.",
    "If you output anything other than a single integer (and absolutely nothing else, including explanatory text), your output will invalidate the dataset. So, please only output a single integer.",
]

header_order = ["orig_examples_header", "imputed_examples_header", "target_example_header", "instructions", "final_words"]
all_headers = [orig_examples_headers, imputed_examples_headers, target_examples_headers, instructions, final_words]
assert len(header_order) == len(all_headers), f"Length of header_order ({len(header_order)}) does not match length of all_headers ({len(all_headers)})"

def get_annotation_value(dataset, annotator, example):
    return round(dataset[example, annotator])

def format_examples(annotator_id, example_ids, imputed=False):
    ans = ""
    for example_count, example_idx in enumerate(example_ids):
        ans += f"Example {example_count + 1}:\n"
        ans += f"Text: {texts[example_idx][0]}\n"
        load_dataset = imputed_annotations if imputed else orig_annotations
        # round the annotation to the nearest integer, so it doesn't show up as a float (we want 0 not 0.0)
        annotation = get_annotation_value(load_dataset, annotator_id, example_idx)
        assert annotation != -1, f"Annotation for annotator {annotator_id} on example {example_idx} was -1"
        ans += f"Annotation from annotator: {annotation}\n\n"

    ans = ans[:-2] # remove the last newlines
    return ans

def format_target_example(example_idx):
    ans = f"Text: {texts[example_idx][0]}\nAnnotation from annotator:"
    return ans

def get_orig_example_ids(annotator_id, target_example_idx=None):
    example_ids = np.where(orig_annotations[:, annotator_id] != -1)[0]
    if target_example_idx is not None:
        # remove the target example from the list
        example_ids = example_ids[example_ids != target_example_idx]
    return example_ids

def get_examples(annotator_id, target_example_idx, imputed_amt=N_IMPUTED_EXAMPLES, orig_amt=N_ORIG_EXAMPLES, only_orig=False):
    if only_orig:
        total_amt = orig_amt + imputed_amt
        example_ids = get_orig_example_ids(annotator_id, target_example_idx)
        if len(example_ids) > total_amt:
            np.random.seed(SEED)
            example_ids = np.random.choice(example_ids, size=total_amt, replace=False)
        if len(example_ids) < total_amt:
            # we don't have enough examples to support this, so use the same ratio as the original, just with a smaller total
            new_total = len(example_ids)
            orig_amt = int(new_total * (orig_amt / total_amt))
            imputed_amt = new_total - orig_amt

        orig_example_ids = example_ids[:orig_amt]
        imputed_example_ids = example_ids[orig_amt:]
        orig_examples = format_examples(annotator_id, orig_example_ids)
        imputed_examples = format_examples(annotator_id, imputed_example_ids, imputed=False)
    else:
        orig_examples = get_orig_examples(annotator_id, target_example_idx, amount=orig_amt)
        imputed_examples = get_imputed_examples(annotator_id, imputed_amt, target_example_idx)
    return orig_examples, imputed_examples

def get_orig_examples(annotator_id, target_example_idx, amount=N_ORIG_EXAMPLES):
    example_ids = get_orig_example_ids(annotator_id, target_example_idx)
    # sample at most N_ORIG_EXAMPLES examples
    if len(example_ids) > amount:
        np.random.seed(SEED)
        example_ids = np.random.choice(example_ids, size=amount, replace=False)
    orig_examples = format_examples(annotator_id, example_ids)
    return orig_examples

def get_imputed_examples(annotator_id, how_many, target_example_idx):
    # imputed data should be completely full, so just pick numbers at random as long as they aren't the orig examples
    orig_example_ids = get_orig_example_ids(annotator_id)
    imputed_example_ids = set(range(len(imputed_annotations))) - set(orig_example_ids) - set([target_example_idx])
    np.random.seed(SEED)
    if how_many > len(imputed_example_ids):
        print(f"WARNING: Not enough imputed examples to support {how_many} examples, so using {len(imputed_example_ids)} instead.")
        how_many = len(imputed_example_ids)
    imputed_example_ids = np.random.choice(list(imputed_example_ids), size=how_many, replace=False)
    imputed_examples = format_examples(annotator_id, imputed_example_ids, imputed=True)
    return imputed_examples

def get_headers_from_header_ids(header_ids):
    headers = []
    for header_index, header_id in enumerate(header_ids):
        if header_id is None:
            headers.append(None)
        else:
            headers.append(all_headers[header_index][header_id])

    return headers

def id_iterator(given_ids, id_maxes):
    """
    Iterate through a list of IDs, incrementing each non-None ID based on the corresponding maximum value in id_maxes.

    This generator function takes two lists as input:
    - given_ids: A list of non-negative integers and Nones.
    - id_maxes: A list of positive integers, representing the maximum value (exclusive) for each corresponding ID in given_ids.

    The function yields the next state of the given_ids list, incrementing the non-None IDs until the maximum value for each ID is reached.

    Args:
        given_ids (List[Union[int, None]]): A list of non-negative integers and Nones.
        id_maxes (List[int]): A list of positive integers, representing the maximum value (exclusive) for each corresponding ID in given_ids.

    Yields:
        List[Union[int, None]]: The next state of the given_ids list, with non-None IDs incremented based on the corresponding maximum value in id_maxes.
    """
    current_ids = given_ids.copy()
    yield current_ids.copy()

    while True:
        for i in range(len(given_ids)):
            if given_ids[i] is not None:
                if current_ids[i] < id_maxes[i] - 1:
                    current_ids[i] += 1
                    break
                else:
                    current_ids[i] = 0
        else:
            break

        yield current_ids.copy()

# These variables allow you to ignore particular prompt versions that aren't very useful
# put indexes of versions of prompts that you always want to ignore
# for example, if you don't like the 2nd version of the 3rd header, set this to [[], [], [1], [], []]
# prompt_versions_to_ignore = [[0, 1], [], [], [], [0]]
prompt_versions_to_ignore = [[], [], [], [], []]
assert len(prompt_versions_to_ignore) == len(header_order), f"Length of prompt_versions_to_ignore ({len(prompt_versions_to_ignore)}) does not match length of header_order ({len(header_order)})"
# put specific combinations of prompt versions that you want to ignore
# for example, if you want to ignore [1, None, 3, 2, 2] and [1, None, 3, 2, 3], set this to [[1, None, 3, 2, 2], [1, None, 3, 2, 3]]
specific_prompt_versions_to_ignore = []
# if this has any versions at all, it will override all ignores, and these will be the only versions included
specific_prompt_versions_to_include = [[4, None, 0, None, 1], [None, None, 0, None, 0], [None, None, 0, 1, 0], [2, 0, 0, None, None], [None, None, None, None, None], [None, 0, 0, None, None], [0, None, 0, None, 0], [None, None, 0, 1, 1], [1, 0, 0, None, None]]

assert not (specific_prompt_versions_to_include and (specific_prompt_versions_to_ignore or (True in [bool(header_index) for header_index in prompt_versions_to_ignore]))), "You can't have specific_prompt_versions_to_include and [specific_prompt_versions_to_ignore or prompt_versions_to_ignore] at the same time"

for dataset in tqdm(DATASETS, desc="Datasets"):
    outputs_folder = "./data_ablation_outputs"
    orig_annotations_location = f"../../datasets/cleaned/{dataset}_annotations.npy"
    imputed_annotations_location = f"../../datasets/final_imputation_results/ncf_imputation_results/{dataset}_-1_ncf_imputation_-1.npy"
    texts_location = f"../../datasets/cleaned/{dataset}_texts.npy"
    rate_limit_pause = 60 / rate_limits[MODEL]

    # Load the data
    orig_annotations = np.load(orig_annotations_location, allow_pickle=True)
    imputed_annotations = np.load(imputed_annotations_location, allow_pickle=True)
    texts = np.load(texts_location, allow_pickle=True)
    # will round the labels to the nearest integer
    valid_labels = sorted(list(map(int, np.unique(orig_annotations[orig_annotations != -1]))))

    # find the N_ANNOTATORS with the *highest* amount of annotations
    annotator_counts = np.sum(orig_annotations != -1, axis=0)
    annotator_ids = np.argsort(annotator_counts)[-N_ANNOTATORS:]
    print(f"Annotators chosen: {annotator_ids}")
        
    dataset_description = file_to_string(f"../distributional/dataset_descriptions/{dataset}_description.txt")     
    example_costs = []
    answers_by_prompt_choice = {}

    # iterate through each prompt
    for prompt_name in tqdm(PROMPT_NAMES, desc="Prompts", leave=False):
        save_folder = f"data_ablation_outputs/{dataset}/{prompt_name}/"
        make_path(save_folder)
        prompt_location = f"./prompts/{prompt_name}.txt"
        for annotator_index, annotator_id in enumerate(tqdm(annotator_ids, desc="Annotators")):
            for replace_data_bool in tqdm([True, False], desc="Replace Data"):
                replace_data_str = "replace" if replace_data_bool else "no_replace"
                ablation_name = f"{prompt_name}_{replace_data_str}"
                unformatted_prompt = file_to_string(prompt_location)
                format_names = get_format_names(unformatted_prompt)

                # [imputed_header, orig_header, target_header]
                header_ids = [None] * len(header_order)
                for header_index, header_name in enumerate(header_order):
                    if header_name in format_names:
                        # make this header included in the cycled headers
                        header_ids[header_index] = 0

                # get the maximum values for each header ID
                header_id_maxes = [len(headers) for headers in all_headers]

                np.random.seed(SEED)
                target_example_idx = np.random.choice(get_orig_example_ids(annotator_id))
                target_example = format_target_example(target_example_idx)
                target_answer = get_annotation_value(orig_annotations, annotator_id, target_example_idx)
                assert target_answer != -1, f"Annotation for annotator {annotator_id} on example {target_example_idx} was -1. The target example should not be empty."

                orig_examples, imputed_examples = get_examples(annotator_id, target_example_idx, imputed_amt=N_IMPUTED_EXAMPLES, orig_amt=N_ORIG_EXAMPLES, only_orig=replace_data_bool)
                format_dict = {
                    "dataset_description": dataset_description,
                    "orig_examples": orig_examples,
                    "imputed_examples": imputed_examples,
                    "target_example": target_example
                }

                for header_ids in tqdm(id_iterator(header_ids, header_id_maxes), desc="Header IDs", total=len(list(id_iterator(header_ids, header_id_maxes))), leave=False):

                    if specific_prompt_versions_to_include and header_ids not in specific_prompt_versions_to_include:
                        tqdm.write(f"Skipping version {str(header_ids)} because it is not in specific_prompt_versions_to_include.")
                        continue

                    if header_ids in specific_prompt_versions_to_ignore:
                        tqdm.write(f"Skipping version {str(header_ids)} because it is in specific_prompt_versions_to_ignore.")
                        continue
                    
                    for header_index, header_id in enumerate(header_ids):
                        if header_id in prompt_versions_to_ignore[header_index]:
                            tqdm.write(f"Skipping version {str(header_ids)} because header {header_index} with ID {header_id} is in prompt_versions_to_ignore.")
                            continue

                    headers = get_headers_from_header_ids(header_ids)
                    header_dict = {header_name: headers[header_index] for header_index, header_name in enumerate(header_order) if headers[header_index] is not None}

                    format_dict.update(header_dict)

                    prompt = unformatted_prompt.format(**format_dict)
                    tqdm.write(f"Have full prompt (of {prompt_name}) for annotator index {annotator_index + 1} with headers {header_ids}")

                    # save the prompt
                    tqdm.write("Saving the prompt...")
                    start_save_location = f"Prompt_{prompt_name}_Example_{target_example_idx}_Annotator_{annotator_index + 1}_{replace_data_str}_Choices_{header_ids}_"
                    save_prompt_location = os.path.join(save_folder, start_save_location + "prompt.txt")
                    make_path(save_prompt_location)
                    with open(save_prompt_location, 'w', encoding=ENCODING) as f:
                        f.write(prompt)

                    # setup messages for chat
                    messages = [
                        {"role": "user", "content": prompt},
                    ]

                    # save messages
                    save_messages_location = os.path.join(save_folder, start_save_location + "messages.json")
                    with open(save_messages_location, 'w', encoding=ENCODING) as f:
                        json.dump(messages, f, indent=4)

                    # Now, have the model complete the prompt
                    tqdm.write("Asking OpenAI to complete the prompt...")
                    error_count = 0
                    while True:
                        try:
                            response = openai.ChatCompletion.create(
                                model=MODEL,
                                max_tokens=MAX_TOKENS,
                                messages=messages,
                                temperature=TEMPERATURE
                            )
                            break
                        except Exception as e:
                            error_count += 1
                            if error_count > 5:
                                if isinstance(e, openai.error.RateLimitError):
                                    raise Exception("Rate limit exceeded too many times.") from e
                                elif isinstance(e, openai.error.ServiceUnavailableError):
                                    raise Exception("Service unavailable too many times.") from e
                                else:
                                    raise e
                            
                            if isinstance(e, openai.error.RateLimitError):
                                tqdm.write(f"Rate limit exceeded. Pausing for {rate_limit_pause} seconds.")
                            elif isinstance(e, openai.error.ServiceUnavailableError):
                                tqdm.write(f"Service unavailable; you likely paused and resumed. Pausing on our own for {rate_limit_pause} seconds to help reset things and then retrying.")
                            else:
                                tqdm.write(f"Type of error: {type(e)}")
                                tqdm.write(f"Error: {e}")
                                tqdm.write(f"Pausing for {rate_limit_pause} seconds.")
                            time.sleep(rate_limit_pause)
                            continue

                    tqdm.write("OpenAI has completed the prompt.")
                    # save the json response
                    tqdm.write("Saving the response...")
                    response_json_save_location = os.path.join(save_folder, start_save_location + "response.json")
                    make_path(response_json_save_location)
                    with open(response_json_save_location, "w", encoding=ENCODING) as f:
                        json.dump(response, f, indent=4)

                    response_text = response.choices[0].message.content
                    response_text_save_location = os.path.join(save_folder, start_save_location + "response.txt")
                    with open(response_text_save_location, "w", encoding=ENCODING) as f:
                        f.write(response_text)
                    tqdm.write("Response saved.")

                    try:
                        model_answer = int(response_text.strip())
                    except:
                        model_answer = None

                    # save the model answer
                    answer = {
                        "model_answer": model_answer,
                        "target_answer": target_answer,
                        "correct": model_answer == target_answer,
                        "valid": model_answer is not None
                    }

                    answer_save_location = os.path.join(save_folder, start_save_location + "answer.json")
                    with open(answer_save_location, "w", encoding=ENCODING) as f:
                        json.dump(answer, f, indent=4)

                    if ablation_name not in answers_by_prompt_choice:
                        answers_by_prompt_choice[ablation_name] = {}

                    prompt_version = str(header_ids)

                    if str(header_ids) not in answers_by_prompt_choice[ablation_name]:
                        answers_by_prompt_choice[ablation_name][str(header_ids)] = {
                            "true_answers": [],
                            "model_answers": []
                        }

                    answers_by_prompt_choice[ablation_name][prompt_version]["true_answers"].append(target_answer)
                    answers_by_prompt_choice[ablation_name][prompt_version]["model_answers"].append(model_answer if model_answer is not None else -1)
                    answers_by_prompt_choice_save_location = os.path.join(save_folder, "answers_by_prompt_choice.json")
                    with open(answers_by_prompt_choice_save_location, "w", encoding=ENCODING) as f:
                        json.dump(answers_by_prompt_choice, f, indent=4)

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

    # now we're saving data for the whole dataset
    save_folder = f"./data_ablation_outputs/{dataset}"
    make_path(save_folder)

    # save the annotator costs as json
    example_costs = {
        "example_costs": example_costs,
        "total_cost": float(np.sum(example_costs))
    }
    with open(make_path(os.path.join(save_folder, "example_costs.json")), "w", encoding=ENCODING) as f:
        json.dump(example_costs, f, indent=4)

    # compute classification reports
    for ablation_name in answers_by_prompt_choice:
        for prompt_version in answers_by_prompt_choice[ablation_name]:
            true_answers = answers_by_prompt_choice[ablation_name][prompt_version]["true_answers"]
            model_answers = answers_by_prompt_choice[ablation_name][prompt_version]["model_answers"]
            choice_classification_report = classification_report(true_answers, model_answers, output_dict=True, zero_division=0)
            answers_by_prompt_choice[ablation_name][prompt_version]["classification_report"] = choice_classification_report

    # save the classification reports
    with open(answers_by_prompt_choice_save_location, "w", encoding=ENCODING) as f:
        json.dump(answers_by_prompt_choice, f, indent=4)

    # summarize the main stats
    clean_stats1 = {
        ablation_name: {
            prompt_version: {
                "accuracy": answers_by_prompt_choice[ablation_name][prompt_version]["classification_report"]["accuracy"],
                "weighted avg f1-score": answers_by_prompt_choice[ablation_name][prompt_version]["classification_report"]["weighted avg"]["f1-score"],
                "valid answer rate": np.mean([1 if answer != -1 else 0 for answer in answers_by_prompt_choice[ablation_name][prompt_version]["model_answers"]])
            }
            for prompt_version in answers_by_prompt_choice[ablation_name]
        }
        for ablation_name in answers_by_prompt_choice
    }

    with open(make_path(os.path.join(save_folder, "major_stats.json")), "w", encoding=ENCODING) as f:
        json.dump(clean_stats1, f, indent=4)

    clean_stats2 = {}
    for ablation_name in clean_stats1:
        best_acc = float('-inf')
        best_f1 = float('-inf')
        best_valid_answer_rate = float('-inf')
        best_acc_version = None
        best_f1_version = None
        best_valid_answer_rate_version = None
        for prompt_version in clean_stats1[ablation_name]:
            acc = clean_stats1[ablation_name][prompt_version]["accuracy"]
            f1 = clean_stats1[ablation_name][prompt_version]["weighted avg f1-score"]
            valid_answer_rate = clean_stats1[ablation_name][prompt_version]["valid answer rate"]
            if acc > best_acc:
                best_acc = acc
                best_acc_version = prompt_version
            if f1 > best_f1:
                best_f1 = f1
                best_f1_version = prompt_version
            if valid_answer_rate > best_valid_answer_rate:
                best_valid_answer_rate = valid_answer_rate
                best_valid_answer_rate_version = prompt_version
        clean_stats2[ablation_name] = {
            "best accuracy": {
                "value": best_acc,
                "version": best_acc_version
            },
            "best weighted avg f1-score": {
                "value": best_f1,
                "version": best_f1_version
            },
            "best valid answer rate": {
                "value": best_valid_answer_rate,
                "version": best_valid_answer_rate_version,
                "accuracy": answers_by_prompt_choice[ablation_name][best_valid_answer_rate_version]["classification_report"]["accuracy"],
                "weighted avg f1-score": answers_by_prompt_choice[ablation_name][best_valid_answer_rate_version]["classification_report"]["weighted avg"]["f1-score"]
            }
        }

    with open(make_path(os.path.join(save_folder, "summary_stats.json")), "w", encoding=ENCODING) as f:
        json.dump(clean_stats2, f, indent=4)

    clean_stats3 = {
        ablation_name: {
            "best accuracy": clean_stats2[ablation_name]["best accuracy"]["value"],
            "best weighted avg f1-score": clean_stats2[ablation_name]["best weighted avg f1-score"]["value"]
        }
        for ablation_name in clean_stats2
    }

    with open(make_path(os.path.join(save_folder, "simple_stats.json")), "w", encoding=ENCODING) as f:
        json.dump(clean_stats3, f, indent=4)

    print(f"Testing complete for dataset {dataset}.")
    print("Statistics:")
    print(json.dumps(clean_stats3, indent=2))
    print("Total Price")
    print(example_costs["total_cost"])