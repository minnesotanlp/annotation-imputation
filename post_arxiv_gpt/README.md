# post_arxiv_gpt
Each of these scripts are generally similar, and are based on the `individual_vs_aggregate_gpt` script. This README covers:
* `post_arxiv_gpt/individual/power_of_imputation.py`
    * Generates data for Table 8
* `post_arxiv_gpt/individual/header_ablation.py`
    * Generates data for Table 13
* `post_arxiv_gpt/individual/data_ablation.py`
    * Generates data for Table 12
* `post_arxiv_gpt/individual/annotator_tagging.py`
    * Generates data for Table 10
* `post_arxiv_gpt/distributional/main.py`
    * Generates data for Table 11
* `post_arxiv_gpt/aggregate/aggregate_imputation_test.py`
    * Generates data to determine if imputed data helps GPT-3 to make aggregate (rather than individual or distributional) predictions
    * (Not included in paper)

Similar to `individual_vs_aggregate_gpt`, there are no executable `.sh` or `.bat` scripts to run these scripts, and parameters are changed by setting the values of constants within the script itself. The scripts are run via `python3 <script_name>.py`, usually from within their own folder, as there may be relative paths.

Just as in `individual_vs_aggregate_gpt`, there should be a `post_arxiv_gpt/../secrets/openai.env` file that is a `.env` file with the value `OPENAI_API_KEY` set to be your OpenAI API key (`OPENAI_API_KEY=sk-...`), and the value `OPENAI_ORGANIZATION` set to be your OpenAI organization ID (`OPENAI_ORGANIZATION=org-...`).

Here is a list of the constants that can be changed, what values they held for our experiments, and their meaning/impact:
* `rate_limits`
    * Determines how long we wait between calls to the API so as not to go past the rate limit
    * This value should not impact the outcome of the experiment
    * Since ChatGPT is the only model being used, this can be set to `{CHAT_GPT_ID: 75}` (which is what we used). However, this is fairly conservative and you can likely raise it slightly higher without any issue.
* `pricing`
    * _This may change over time depending on OpenAI_
    * If the pricing changes, you can set it here. For our experiments, the pricing was `{GPT4_ID: {PROMPT_PRICE_PER_TOKEN_KEY: 0.03 / 1000, COMPLETION_PRICE_PER_TOKEN_KEY: 0.06 / 1000}, GPT3_ID: {GENERAL_PRICE_PER_TOKEN_KEY: 0.02 / 1000}, CHAT_GPT_ID: {PROMPT_PRICE_PER_TOKEN_KEY: 0.002 / 1000, COMPLETION_PRICE_PER_TOKEN_KEY: 0.002 / 1000}}`. Since only Chat_GPT_ID was used, other pricing may be inaccurate.
* `MODEL`
    * What model is used for completions
        * Only models that support _Chat_ can be used with these scripts, unlike `individual_vs_aggregate_gpt` (which used Completions, not Chat).
    * Set to `Chat_GPT_ID`
* `DATASETS`
    * List of the dataset names you'd like to run the script on
    * While we ran our experiments in smaller groups, to run all of our datasets, `DATASETS` should be set to `["ghc", "SChem", "Sentiment", "SChem5Labels", "politeness", "Fixed2xSBIC"]`
* `MAX_TOKENS`
    * How many tokens the model will output at maximum.
        * If your dataset has labels with more than 2 digits (more than 99 labels), you may need to increase this value. However, increasing it also increases the chance that extra symbols are caught, invalidating the model's output. (Any outputs that are not a valid label are considered invalid.)
    * We set this to `2` for our experiments
* `SEED`
    * A (fixed) random number, used to facilitate determinism when choosing samples and annotators.
    * This was set to `29` for our experiments.
* `N_IMPUTED_EXAMPLES`
    * How many imputed samples to include in the prompt
        * Prompts will include exactly this many imputed samples in each prompt (unless the prompt does not include any imputed samples)
    * Set to `30`
* `N_ORIG_EXAMPLES`
    * How many original/non-imputed samples to include in the prompt
        * Prompts may include less than this amount if there are not enough available, but will never include more than this amount
    * Set to `30`
* `N_ANNOTATORS`
    * Number of annotators whose annotations will be predicted.
        * This is usually one of the main bottlenecks of the code, as we will test this many annotators for each of the prompt combinations selected.
    * Set to `30`
* `TEMPERATURE`
    * How random the AI's outputs will be.
    * This should always set to `0` for reproducibility
* `ENCODING`
    * The encoding used for text
    * Should always be set to `"utf-8"`
* `save_folder`
    * _Only used for distributional and aggregate scripts._
    * Where the outputs should be saved to. Usually includes a description of what's being run.
    * Here's what these should be set to for the different scripts:
        * distributional
            * `f"outputs/{DATASET}_no_context_prompt/"`
                * (Assuming you are using the better-performing "no-context" prompt that is what is used for the results in the paper)
        * aggregate
            * `f"outputs/{dataset}/{prompt_name}/{swapping_headers_str}/{swapping_data_str}/{replacing_data_str}/{which_data_replaced_str}/"`
        * (Others)
            * (Should automatically be set within the main loop)
* `prompt_location`
    * _Only used in distributional script._
    * The location of the prompt that should be used.
        * (This script only uses a single prompt (without variations).)
    * For our experiments, this was set to `"./prompts/no_context_prompt.txt"`
* `orig_annotations_location`
    * Where to get the original annotations from
    * For our experiments, this was set to `f"../../datasets/cleaned/{dataset}_annotations.npy"`
        * Capitalize `dataset` to `DATASET` for `individual/annotator_tagging.py` and `distributional/main.py`
* `imputed_annotations_location`
    * _This is what decides which imputation method is used_
    * Where to get the imputed annotations from
    * For our experiments, this was set to `f"../../datasets/final_imputation_results/ncf_imputation_results/{dataset}_-1_ncf_imputation_-1.npy"`
        * (We only used NCF imputation)
        * Capitalize `dataset` to `DATASET` for `individual/annotator_tagging.py` and `distributional/main.py`
* `texts_location`
    * Where to get the texts from
    * For our experiments, this was set to `f"../../datasets/cleaned/{dataset}_texts.npy"`
        * Capitalize `dataset` to `DATASET` for `individual/annotator_tagging.py` and `distributional/main.py`
* `ablations_to_include`
    * _Only used in aggregate script._
    * Determines which ablation studies should be done
        * This script essentially combines the data ablation and header ablation study from the individual prediction scripts into one singular script that both tests and does the ablation studies.
        * See the code for details about how to set this properly
    * Could be set to:
        ```
        [
            [False, False, False, False], # initial
            [False, False, True, False], # replacing imputed data with original data
            [False, True, False, False], # swapping the position of the imputed and original data
        ]
        ```
* `PROMPT_NAMES`
    * The names of the prompts that will be tested
    * For our experiments, this is set to `["just_orig7", "just_orig6", "just_orig5", "combined3", "just_imputed1", "just_imputed3"]`
        * You can add your own prompts to the `prompt` folder, and then add them here as well. However, each new prompt may add a significant amount of time to the script, as scripts generally test an exponential number of variations of the same prompt (each possible combination of headers).
* `imputed_examples_headers`
    * The list of potential options for headers used to preface the imputed examples
    * For our experiments, this is set to:
        ```
        [
            "Estimated Examples:",
            "Imputed Examples:"
        ]
        ```
* `orig_examples_headers`
    * The list of potential options for headers used to preface the original/non-imputed examples
    * For our experiments, this is set to:
        ```
        [
            "Correct Examples:",
            "Human-rated Examples:",
            "Human-Rated Examples:",
            "Human-rated Examples: (there may not be any)",
            "Examples from the dataset (there may not be any):"
            "Examples from the dataset (there may not be any)",
        ]
        ```
* `target_examples_headers`
    * The list of potential options for headers used to preface the target example text that should be labeled by the AI
    * Set to:
        ```
        [
            "Target Example:"
        ]
        ```
* `instructions``
    * List of potential instructions for the AI on how to label the example
    * Set to:
        ```
        [
            "Predict the integer label of the following example:",
            "Now you will make your prediction (if you are unsure, just give your best estimate.) Your output should be an integer label:"
        ]
        ```
* `final_words`
    * Final text usually intented to make it more likely that the AI system conforms to the correct format
    * Set to:
        ```
        [
            "Your output should be a single integer corresponding to the label.",
            "Your output should be a single integer and nothing else.",
            "The only valid output is a single integer.",
            "If you output anything other than a single integer, your output will be considered invalid."
            "If you output anything other than a single integer, your output will harm the integrity of our dataset.",
            "If you output anything other than a single integer (and absolutely nothing else, including explanatory text), your output will invalidate the dataset.",
            "If you output anything other than a single integer (and absolutely nothing else, including explanatory text), your output will invalidate the dataset. So, please only output a single integer.",
        ]
        ```
* `prompt_versions_to_ignore`
    * Enables you to put indexes of versions of prompts that you don't want to run.
        * For example, if you don't like the 2nd version of the 3rd header, set this to `[[], [], [1], [], []]`
        * Note that the order of headers (which slot refers to which header) is determined by the `header_order` variable
    * For our experiments, this is set to `[[], [], [], [], []]` (nothing is ignored)
* `specific_prompt_versions_to_ignore`
    * Enables the user disable particular combinations of prompt versions.
        * For example if you want to specifically ignore [1, None, 3, 2, 2] (1st version of 1st prompt, 2nd prompt doesn't exist for this skeleton, 3nd version of 3rd prompt, 2nd version of 4th prompt, 2nd version of 5th prompt) and [1, None, 3, 2, 3], you would set this to `[[1, None, 3, 2, 2], [1, None, 3, 2, 3]]`
    * For our experiments, this is set to `[]` (nothing is ignored)
* `specific_prompt_versions_to_include`
    * _This will override all ignores_
    * Allows you to pick particular prompt version combinations to include in the test.
    * For our experiments, this was set to: `[[4, None, 0, None, 1], [None, None, 0, None, 0], [None, None, 0, 1, 0], [2, 0, 0, None, None], [None, None, None, None, None], [None, 0, 0, None, None], [0, None, 0, None, 0], [None, None, 0, 1, 1], [1, 0, 0, None, None]]`
        * These were handpicked based on a small initial test of valid and correct response rates (not on whether they supported our hypothesis)
            * Other prompt variations are more likely to result in invalid and incorrect responses. Essentially we picked any combination of prompts that seemed to do decently on any of the datasets.
        * When reporting results, we report the result from the best-performing prompt and ignore others, essentially performing a grid-search over the prompt space.


To use a new dataset, you'll need to describe the dataset in `post_arxiv_gpt/distributional/dataset_descriptions`, as these scripts look up dataset descriptions from that location.

To use the results, analyze the `summary_stats.json` (or `clean_statistics.json`) file, and look for the best result among the prompts that fits your criteria. (Note that `clean_statistics.json` files tend to only contain the statistics you want, and do not require this extra parsing.) For example, if you are attempting to replicate the data ablation study (Table 12), and you want to determine the result for the case of SChem with just original data that has a header claiming that the data is imputed (i.e. the All Original/Imputed/SChem box), you would look for the highest F1 score among the prompts labeled `just_imputed` (since the prompt is for just imputed data) with the suffix `replace` (since the data has been replaced with original data). Using our experiment settings, there should be results for `just_imputed1_replace` and `just_imputed3_replace`, and we report the better of the two. There are other statistics files which go into more details about the performance of particular versions of the prompts as well. If it is unclear which outputs refer to what, please create a GitHub issue describing what you are trying to do and/or email lowma016@umn.edu.

In addition, each of these scripts generally has built-in mechanisms to deal with sporadic errors from the OpenAI API and will retry multiple times before giving up.