# individual_vs_aggregate_gpt/main.py
This script runs the experiment for Table 9

Unfortunately, it does not have an easy-to-use shell script with options.
Instead, it must be run with `python3 main.py`, and the options are chosen by setting constants to different values.

Here's a list of the constants and how they should be handled:
* `MODEL`
    * Indicates which OpenAI model should be used.
        * (Note that while GPT-4 is listed, at the time of writing, GPT-4 only supports the Chat API, which is not compatible with this code. This code relies on the Completion API.)
    * Should be set to `GPT3_ID` to replicate our results.
* `DATASET`
    * The name of the dataset
* `DATASET_LOCATION`
    * Should be set to a format string such that when `{dataset}` is replaced with the name of the dataset (i.e. the value of `DATASET`) and `{annotations_or_texts}` is replaced with either the string "annotations" or "texts", the string will then be a valid location of either the annotations for the dataset, or the texts for the dataset, respectively.
* `IMPUTED_DATASET_LOCATION`
    * Same as `DATASET_LOCATION` except for that it should always point to the imputed annotations (only the `{dataset}` placeholder is used)
        * We used the imputation from the NCF imputer
* `N_SHOTS`
    * If doing individual shots, how many samples from each individual should be put into the prompt? (The script will not include any annotators who have less than N_SHOTS + 1 responses. The script finds a pool of valid annotators, and each annotator must have N_SHOTS of their own responses, plus 1 extra as the test sample.)
    * This was set to `4` for our experiments.
* `N_ANNOTATORS`
    * How many annotators to include in the experiment.
    * Here are the values this was set to for different datasets:
        * SChem: 20
        * SBIC: 30
        * GHC: 18
* `MAX_TOKENS`
    * Max tokens for GPT-3 to output.
    * This was set to `5` for our experiments.
* `SEED`
    * A (fixed) random number, used to facilitate determinism when choosing samples and annotators.
    * This was set to `47` for our experiments.
* `k_shots` (lower down)
    * **This determines what type of data is used for the experiment: imputed data, original data, or majority-voted data.**
    * This can be set to any one of:
        * `ExtraDataType.ORIGINAL_DISTRIBUTIONAL` for original distributional data
        * `ExtraDataType.IMPUTED_DISTRIBUTIONAL` for imputed distributional data
        * `ExtraDataType.MAJORITY_VOTED` for majority voted data
            * (Previously, this was indicated by setting it to `-1`, so if you see that value in the output files, that is what it referred to.)
        * A positive integer for that number of shots
        * `ExtraDataType.ALL_SHOTS` for all possible shots
            * (Previously, this was indicated by setting it to `None`, so if you see that value in the output files, that is what it referred to.)
    * For our experiments, we used the original distributional data, imputed distributional data, and majority voted data options (running the script 3 times), all of which are reported in Table 9

Also ensure that `../secrets/openai.env` is a `.env` file with the value `OPENAI_API_KEY` set to be your OpenAI API key (`OPENAI_API_KEY=sk-...`), and the value `OPENAI_ORGANIZATION` set to be your OpenAI organization ID (`OPENAI_ORGANIZATION=org-...`).

Once all these values are set to your liking, you can run `python3 main.py`, and the `output` folder will fill up with all of the outputs from GPT-3. Costs will be saved in `annotator_costs.json`, and the data to fill in Table 9 can be found in the resulting `statistics.json`. (You can find our outputs in the `output` folder.)