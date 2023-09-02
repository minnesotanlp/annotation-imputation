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
import itertools
from sklearn.metrics import classification_report
import scipy

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
rate_limits = {GPT3_ID: 20, GPT4_ID: 200, CHAT_GPT_ID: 75}

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
DATASETS = ["ghc", "politeness", "Fixed2xSBIC", "SChem", "Sentiment", "SChem5Labels"]
# DATASETS = ['SChem']
MAX_TOKENS = 2
SEED = 29
N_ORIG_EXAMPLES = [10]
N_IMPUTED_EXAMPLES = [10]
N_TARGET_EXAMPLES = [20]
TEMPERATURE = 0
ENCODING = "utf-8"

assert len(N_IMPUTED_EXAMPLES) == len(N_ORIG_EXAMPLES), f"Must have the same number of imputed and original example sets, but got {len(N_IMPUTED_EXAMPLES)} imputed example sets and {len(N_ORIG_EXAMPLES)} original example sets"
assert len(N_TARGET_EXAMPLES) == len(N_ORIG_EXAMPLES), f"Must have the same number of target example sets and original example sets, but got {len(N_TARGET_EXAMPLES)} target example sets and {len(N_ORIG_EXAMPLES)} original example sets"

# set the seed
np.random.seed(SEED)

# PROMPT_NAMES = ["combined1", "just_orig1", "just_imputed1", "combined3", "just_imputed3", "just_orig3"]
PROMPT_NAMES = ["aggregate_combined1"]

# make sure that all prompts exist so it doesn't crash later
for prompt_name in PROMPT_NAMES:
    assert os.path.exists(f"./prompts/{prompt_name}.txt"), f"Prompt {prompt_name} does not exist (could not be found at {os.path.abspath(f'./prompts/{prompt_name}.txt')})"

imputed_examples_headers = [
    "",
    # "More Examples:\n",
    # "Estimated Examples:\n",
    # "Imputed Examples:\n",
    "Estimated Majority-voted Examples:\n",
    "Imputed Majority-voted Examples:\n",
    # "Estimated Aggregated Examples:\n",
    # "Imputed Aggregated Examples:\n",
]

orig_examples_headers = [
    "",
    # "Examples:\n",
    "Human-Labeled Majority-voted Examples:\n",
]

target_examples_headers = [
    "Target Example:"
]

instructions = [
    "Predict the integer label of the following example:\n",
    "Now you will make your prediction (if you are unsure, just give your best estimate.) Your output should be an integer label:\n"
]

final_words = [
    "",
    # "Your output should be a single integer corresponding to the label.\n",
    "Your output should be a single integer and nothing else.\n",
    # "The only valid output is a single integer.\n",
    # "If you output anything other than a single integer, your output will be considered invalid.\n",
    # "If you output anything other than a single integer, your output will harm the integrity of our dataset.\n",
    # "If you output anything other than a single integer (and absolutely nothing else, including explanatory text), your output will invalidate the dataset.\n",
    "If you output anything other than a single integer (and absolutely nothing else, including explanatory text), your output will invalidate the dataset. So, please only output a single integer.\n",
]

header_order = ["orig_examples_header", "imputed_examples_header", "target_example_header", "instructions", "final_words"]
all_headers = [orig_examples_headers, imputed_examples_headers, target_examples_headers, instructions, final_words]
assert len(header_order) == len(all_headers), f"Length of header_order ({len(header_order)}) does not match length of all_headers ({len(all_headers)})"

def get_majority_vote(dataset, example_idx):
    annotations = dataset[example_idx]
    # replace -1 with nan so that it doesn't affect the mode
    annotations = np.where(annotations == -1, np.nan, annotations)
    maj_vote, _maj_vote_count = scipy.stats.mode(annotations, nan_policy='omit', axis=None, keepdims=False)
    assert maj_vote == int(maj_vote), f"Majority vote was not an integer: {maj_vote}"
    maj_vote = int(maj_vote)
    return maj_vote

def get_annotation_value(dataset, annotator, example):
    return round(dataset[example, annotator])

def format_examples(example_ids, dataset):
    ans = ""
    for example_count, example_idx in enumerate(example_ids):
        ans += f"Example {example_count + 1}:\n"
        ans += f"Text: {texts[example_idx][0]}\n"
        # round the annotation to the nearest integer, so it doesn't show up as a float (we want 0 not 0.0)
        annotation = get_majority_vote(dataset, example_idx)
        assert annotation != -1, f"Majority-voted annotation for example {example_idx} was -1"
        ans += f"Annotation: {annotation}\n\n"

    ans = ans[:-2] # remove the last newlines
    return ans

# def format_examples(annotator_id, example_ids, imputed=False):
#     ans = ""
#     for example_count, example_idx in enumerate(example_ids):
#         ans += f"Example {example_count + 1}:\n"
#         ans += f"Text: {texts[example_idx][0]}\n"
#         load_dataset = imputed_annotations if imputed else orig_annotations
#         # round the annotation to the nearest integer, so it doesn't show up as a float (we want 0 not 0.0)
#         annotation = get_annotation_value(load_dataset, annotator_id, example_idx)
#         assert annotation != -1, f"Annotation for annotator {annotator_id} on example {example_idx} was -1"
#         ans += f"Annotation from annotator: {annotation}\n\n"

#     ans = ans[:-2] # remove the last newlines
#     return ans

def format_target_example(example_idx):
    ans = f"Text: {texts[example_idx][0]}\nAnnotation:"
    return ans

# These are done in the main loop code now
# def get_orig_examples(annotator_id, target_example_idx):
#     example_ids = get_orig_example_ids(annotator_id, target_example_idx)
#     # sample at most N_ORIG_EXAMPLES examples
#     if len(example_ids) > N_ORIG_EXAMPLES:
#         np.random.seed(SEED)
#         example_ids = np.random.choice(example_ids, size=N_ORIG_EXAMPLES, replace=False)
#     orig_examples = format_examples(annotator_id, example_ids)
#     return orig_examples

# def get_imputed_examples(annotator_id, how_many, target_example_idx):
#     # imputed data should be completely full, so just pick numbers at random as long as they aren't the orig examples
#     format_examples(annotator_id, imputed_example_ids, imputed=True)
#     return imputed_examples

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
# specific_prompt_versions_to_include = [[4, None, 0, None, 1], [None, None, 0, None, 0], [None, None, 0, 1, 0], [2, 0, 0, None, None], [None, None, None, None, None], [None, 0, 0, None, None], [0, None, 0, None, 0], [None, None, 0, 1, 1], [1, 0, 0, None, None]]
specific_prompt_versions_to_include = []

# if empty, include all ablations
# [swapping_headers, swapping_data, replacing_data, which_data_replaced]
ablations_to_include = [
    [False, False, False, False], # initial
    [False, False, True, False], # replacing imputed data with original data
    [False, True, False, False], # swapping the position of the imputed and original data
]

for ablation in ablations_to_include:
    assert isinstance(ablation, list), "ablations_to_include must be a list of lists"

assert not (specific_prompt_versions_to_include and (specific_prompt_versions_to_ignore or (True in [bool(header_index) for header_index in prompt_versions_to_ignore]))), "You can't have specific_prompt_versions_to_include and [specific_prompt_versions_to_ignore or prompt_versions_to_ignore] at the same time"

for dataset in tqdm(DATASETS, desc="Datasets"):
    outputs_folder = "./outputs"
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

    get_orig_examples = lambda idxs: format_examples(idxs, orig_annotations)
    get_imputed_examples = lambda idxs: format_examples(idxs, imputed_annotations)

    dataset_description = file_to_string(f"../distributional/dataset_descriptions/{dataset}_description.txt")     
    
    example_costs = []
    # store the response based on the ablation and headers
    answers_by_prompt_choice = {}
    # iterate through each prompt
    for prompt_name in tqdm(PROMPT_NAMES, desc="Prompts", leave=False):
        prompt_location = f"./prompts/{prompt_name}.txt"

        # iterate through number of examples
        for num_examples_index in tqdm(range(len(N_ORIG_EXAMPLES)), desc="Number of examples", leave=False):
            n_orig_examples = N_ORIG_EXAMPLES[num_examples_index]
            n_imputed_examples = N_IMPUTED_EXAMPLES[num_examples_index]
            n_target_examples = N_TARGET_EXAMPLES[num_examples_index]

            target_example_idxs = np.random.choice(len(orig_annotations), n_target_examples, replace=False)
            
            # iterate through each target example
            for target_example_idx in tqdm(target_example_idxs, desc="Target examples", leave=False):
                # get random indices for the target, original, and imputed examples
                remaining_idxs = np.arange(len(orig_annotations))
                remaining_idxs = np.delete(remaining_idxs, target_example_idx)
                np.random.seed(SEED)
                orig_idxs = np.random.choice(remaining_idxs, n_orig_examples, replace=False)
                remaining_idxs = np.delete(remaining_idxs, orig_idxs)
                np.random.seed(SEED)
                imputed_idxs = np.random.choice(remaining_idxs, n_imputed_examples, replace=False)

                # iterate through which ablation study we're doing
                # we just ignore which_data_replaced if replacing_data is False
                len_ablation_study = 2 ** 4
                for swapping_headers, swapping_data, replacing_data, which_data_replaced in tqdm(itertools.product([True, False], repeat=4), desc="Ablation study", total=len_ablation_study, leave=False):

                    ablation = [swapping_headers, swapping_data, replacing_data, which_data_replaced]
                    if not ablations_to_include or ablation not in ablations_to_include:
                        tqdm.write(f"Skipping ablation {ablation}")
                        continue

                    if not replacing_data and not which_data_replaced:
                        # if we've already done this ablation, skip it
                        if [swapping_headers, swapping_data, replacing_data, True] in ablations_to_include or not ablations_to_include:
                            tqdm.write(f"We already did this ablation since when not replacing data, which_data_replaced is ignored. Skipping ablation {ablation}")
                            continue

                    swapping_headers_str = "headers_swap" if swapping_headers else "no_headers_swap"
                    swapping_data_str = "data_swap" if swapping_data else "no_data_swap"
                    replacing_data_str = "replaced" if replacing_data else "no_replace"
                    if replacing_data:
                        which_data_replaced_str = "1st_replaced" if which_data_replaced else "2nd_replaced"
                    else:
                        which_data_replaced_str = "none_replaced"

                    prompt_ablation = f"{swapping_headers_str}_{swapping_data_str}_{replacing_data_str}_{which_data_replaced_str}"
                    

                    save_folder = f"outputs/{dataset}/{prompt_name}/{swapping_headers_str}/{swapping_data_str}/{replacing_data_str}/{which_data_replaced_str}/"
                    answers_by_prompt_choice_save_location = os.path.join(save_folder, "answers_by_prompt_choice.json")
                    make_path(save_folder)

                    # get the prompt
                    unformatted_prompt = file_to_string(prompt_location)
                    format_names = get_format_names(unformatted_prompt)

                    # get the examples
                    if swapping_data:
                        # it's the imputed that come first now
                        get_examples1_func = get_imputed_examples
                        get_examples1_data = imputed_idxs
                        get_examples2_func = get_orig_examples
                        get_examples2_data = orig_idxs
                    else:
                        # orig then imputed, as usual
                        get_examples1_func = get_orig_examples
                        get_examples1_data = orig_idxs
                        get_examples2_func = get_imputed_examples
                        get_examples2_data = imputed_idxs

                    if replacing_data:
                        if which_data_replaced:
                            # Make the first examples the same kind as the second
                            get_examples1_func = get_examples2_func
                        else:
                            get_examples2_func = get_examples1_func

                    orig_examples = get_examples1_func(get_examples1_data)
                    imputed_examples = get_examples2_func(get_examples2_data)
                    
                    # get the target example
                    target_example = format_target_example(target_example_idx)
                    target_answer = get_majority_vote(orig_annotations, target_example_idx)

                    # get the headers
                    # [imputed_header, orig_header, target_header]
                    header_ids = [None] * len(header_order)
                    for header_index, header_name in enumerate(header_order):
                        if header_name in format_names:
                            # make this header included in the cycled headers
                            header_ids[header_index] = 0

                    # get the maximum values for each header ID
                    header_id_maxes = [len(headers) for headers in all_headers]

                    format_dict = {
                        "dataset_description": dataset_description,
                        "orig_examples": orig_examples,
                        "imputed_examples": imputed_examples,
                        "target_example": target_example
                    }

                    # iterate through different header fillers
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

                        header_ids_str = str(header_ids)

                        headers = get_headers_from_header_ids(header_ids)
                        header_dict = {header_name: headers[header_index] for header_index, header_name in enumerate(header_order) if headers[header_index] is not None}

                        format_dict.update(header_dict)

                        if swapping_headers:
                            # swap "orig_examples_header" and "imputed_examples_header"
                            orig_examples_header = format_dict["orig_examples_header"]
                            imputed_examples_header = format_dict["imputed_examples_header"]
                            format_dict["orig_examples_header"] = imputed_examples_header
                            format_dict["imputed_examples_header"] = orig_examples_header

                        prompt = unformatted_prompt.format(**format_dict)

                        # save the prompt
                        tqdm.write("Saving the prompt...")
                        start_save_location = f"Prompt_{prompt_name}_Example_{target_example_idx}_NOrig_{n_orig_examples}_NImput_{n_imputed_examples}_{swapping_headers_str}_{swapping_data_str}_{replacing_data_str}_{which_data_replaced_str}_Choices_{header_ids}_"
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
                            print(answer)
                            json.dump(answer, f, indent=4)

                        if prompt_ablation not in answers_by_prompt_choice:
                            answers_by_prompt_choice[prompt_ablation] = {}

                        header_ids_str = str(header_ids)
                        prompt_version = header_ids_str

                        if str(header_ids) not in answers_by_prompt_choice[prompt_ablation]:
                            answers_by_prompt_choice[prompt_ablation][str(header_ids)] = {
                                "true_answers": [],
                                "model_answers": []
                            }

                        answers_by_prompt_choice[prompt_ablation][prompt_version]["true_answers"].append(target_answer)
                        answers_by_prompt_choice[prompt_ablation][prompt_version]["model_answers"].append(model_answer if model_answer is not None else -1)
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
    big_save_folder = f"./outputs/{dataset}"
    make_path(big_save_folder)

    # save the annotator costs as json
    example_costs = {
        "example_costs": example_costs,
        "total_cost": float(np.sum(example_costs))
    }
    with open(make_path(os.path.join(big_save_folder, "example_costs.json")), "w", encoding=ENCODING) as f:
        json.dump(example_costs, f, indent=4)

    # compute classification reports
    for prompt_ablation in answers_by_prompt_choice:
        for prompt_version in answers_by_prompt_choice[prompt_ablation]:
            true_answers = answers_by_prompt_choice[prompt_ablation][prompt_version]["true_answers"]
            model_answers = answers_by_prompt_choice[prompt_ablation][prompt_version]["model_answers"]
            choice_classification_report = classification_report(true_answers, model_answers, output_dict=True, zero_division=0)
            answers_by_prompt_choice[prompt_ablation][prompt_version]["classification_report"] = choice_classification_report

    # save the classification reports
    with open(answers_by_prompt_choice_save_location, "w", encoding=ENCODING) as f:
        json.dump(answers_by_prompt_choice, f, indent=4)

    # summarize the main stats
    clean_stats1 = {
        prompt_ablation: {
            prompt_version: {
                "accuracy": answers_by_prompt_choice[prompt_ablation][prompt_version]["classification_report"]["accuracy"],
                "weighted avg f1-score": answers_by_prompt_choice[prompt_ablation][prompt_version]["classification_report"]["weighted avg"]["f1-score"],
                "valid answer rate": np.mean([1 if answer != -1 else 0 for answer in answers_by_prompt_choice[prompt_ablation][prompt_version]["model_answers"]])
            }
            for prompt_version in answers_by_prompt_choice[prompt_ablation]
        }
        for prompt_ablation in answers_by_prompt_choice
    }

    with open(make_path(os.path.join(big_save_folder, "major_stats.json")), "w", encoding=ENCODING) as f:
        json.dump(clean_stats1, f, indent=4)

    clean_stats2 = {}
    for prompt_ablation in clean_stats1:
        best_acc = float('-inf')
        best_f1 = float('-inf')
        best_valid_answer_rate = float('-inf')
        best_acc_version = None
        best_f1_version = None
        best_valid_answer_rate_version = None
        for prompt_version in clean_stats1[prompt_ablation]:
            acc = clean_stats1[prompt_ablation][prompt_version]["accuracy"]
            f1 = clean_stats1[prompt_ablation][prompt_version]["weighted avg f1-score"]
            valid_answer_rate = clean_stats1[prompt_ablation][prompt_version]["valid answer rate"]
            if acc > best_acc:
                best_acc = acc
                best_acc_version = prompt_version
            if f1 > best_f1:
                best_f1 = f1
                best_f1_version = prompt_version
            if valid_answer_rate > best_valid_answer_rate:
                best_valid_answer_rate = valid_answer_rate
                best_valid_answer_rate_version = prompt_version
        clean_stats2[prompt_ablation] = {
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
                "accuracy": answers_by_prompt_choice[prompt_ablation][best_valid_answer_rate_version]["classification_report"]["accuracy"],
                "weighted avg f1-score": answers_by_prompt_choice[prompt_ablation][best_valid_answer_rate_version]["classification_report"]["weighted avg"]["f1-score"]
            }
        }

    with open(make_path(os.path.join(big_save_folder, "summary_stats.json")), "w", encoding=ENCODING) as f:
        json.dump(clean_stats2, f, indent=4)

    clean_stats3 = {
        prompt_ablation: {
            "best accuracy": clean_stats2[prompt_ablation]["best accuracy"]["value"],
            "best weighted avg f1-score": clean_stats2[prompt_ablation]["best weighted avg f1-score"]["value"]
        }
        for prompt_ablation in clean_stats2
    }

    with open(make_path(os.path.join(big_save_folder, "simple_stats.json")), "w", encoding=ENCODING) as f:
        json.dump(clean_stats3, f, indent=4)

    print(f"Testing complete for dataset {dataset}.")
    print("Statistics:")
    print(json.dumps(clean_stats3, indent=2))
    print("Total Price")
    print(example_costs["total_cost"])