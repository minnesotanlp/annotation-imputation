This repository contains code for the paper
**"Annotation Imputation to Individualize Predictions: Initial Studies on Distribution Dynamics and Model Predictions"** 
by London Lowmanstone (lowma016@umn.edu), Ruyuan Wan, Risako Owan, [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), and [Dongyeop Kang](https://dykang.github.io/).

If you have any issues running this project, please create a GitHub issue, and if it's not addressed, please also email lowma016@umn.edu. We really care about making sure that our results are reproducible and that our methods are helpful for understanding the impacts on new datasets and imputation methods.

We use 6 different datasets in our paper. The `.zip` file containing all of our datasets can be found at https://drive.google.com/file/d/1dir3CxFoHkO1b4eviN5onbEwOiwTtgqE/view?usp=sharing and should be loaded into the `/datasets` folder.

This project consists of different scripts for each portion of our paper. Here is an overview of each of the scripts and what they do. Python scripts may also have associated `.bat` (for Windows) or `.sh` (for Linux) scripts in their folder. These executable scripts may be used to run the Python script how we ran it, or to better understand the parameters of the script for easy adaptation to your own datasets and imputation methods.

The requirements for the code can be handled by using the Docker image located at https://registry.hub.docker.com/r/londonl/nlperspectives to ensure functionality. You can also find the full requirements at `requirements/requirements.txt` (the Docker image is merely a light wrapper around these requirements). The container should be launched with a command such as `docker run -it --gpus all --volume ~/annotation-imputation:/workspace londonl/nlperspectives` where `~/annotation-imputation` is this cloned GitHub repo.

## How to Replicate Our Paper
* Figure 1
    * N/A
* Figure 2
    * N/A
* Table 1
    * Read the original papers for each dataset
    * Look at the values in the original datasets or our cleaned versions of them
* Table 2
    * Clean the datasets (see Data Cleaning Scripts below)
    * Run the `utilities/test_imputer_rmse` script
        * (No imputation needs to be done beforehand - the script should handle everything.)
* Figure 3
    * Clean the datasets (see Data Cleaning Scripts below)
    * Run the Multitask (`multitask`), NCF (`ncf_matrix_factorization`), and Kernel (`kernel_matrix_factorization`) imputers on the cleaned datasets
    * Use the resulting imputed datasets (and the original dataset) and run them through the `utilities/pca_plots` script
* Table 3
    * Clean the datasets (see Data Cleaning Scripts below)
    * Run the NCF imputer (`ncf_matrix_factorization`) on the cleaned datasets
    * Run the `utilities/variance_disagreement_graphs` script to compare the original and NCF-imputed datasets
        * Use the output json data to recreate the table
* Figure 4
    * Clean the datasets (see Data Cleaning Scripts below)
    * Run the NCF imputer (`ncf_matrix_factorization`) on the cleaned datasets
    * Run the `utilities/variance_disagreement_graphs` script to compare the original and NCF-imputed datasets
        * The script will output the figure
* Table 4
    * Clean the datasets (see Data Cleaning Scripts below)
    * Run the Multitask (`multitask`), NCF (`ncf_matrix_factorization`), and Kernel (`kernel_matrix_factorization`) imputers on the cleaned datasets
    * Compute the KL Divergence using `disagreement_shift/label_distribution`
    * Use the KL Divergence to run `disagreement_shift/compare_kl` to compare different imputers, which will also contain the mean and standard deviation of the KL across all examples
        * Use these results to recreate the table
* Table 5
    * Clean the datasets (see Data Cleaning Scripts below)
    * Run the Multitask (`multitask`) and NCF (`ncf_matrix_factorization`) imputers on the cleaned datasets
    * Rerun the Multitask imputer twice: 1. Train it on the output of the first Multitask run 2. Train it on the output of the NCF imputer
        * Use the F1 Score from the final epoch (as recorded on Weights and Bases or Tensorboard if you choose to use that). Compute the mean and standard deviation across folds by hand.
* Figure 5
    * Clean the datasets (see Data Cleaning Scripts below)
    * Run the NCF (`ncf_matrix_factorization`) and Kernel (`kernel_matrix_factorization`) imputers on the cleaned datasets
    * Use `disagreement_shift/label_distribution` to compute the KL divergence values between the original and imputed data. The same script will also compute the distributional/soft labels in the process.
    * Use the KL divergence and soft label outputs from the previous step, rename them correctly (as described in `disagreement_shift/website/README.md`), and then run the `website` script which will launch the site.
    * Visit the site, which will automatically download the outputs.
* Table 6
    * See the instructions for replicating Table 5, but use the aggregate rather than individual statistics.
* Table 7
    * Clean the datasets (see Data Cleaning Scripts below)
    * Run the Multitask imputer (`multitask`) on the cleaned datasets
        * This will output a `.json` file with the final predictions
    * Use the `.json` output with the predictions together with the original dataset to run the `split_based_on_disagreement` script, which will output all the data for the table
* Figure 6 / Appendix G
    * See the steps for replicating Figure 3
* Figure 7 / Appendix H
    * See the steps for replicating Figure 4
* Table 8
    * Clean the datasets (see Data Cleaning Scripts below)
    * Run the NCF imputer (`ncf_matrix_factorization`) on the cleaned datasets
    * Run the `post_arxiv_gpt/individual/power_of_imputation.py` script
    * Analyze the outputs of the script as described in `post_arxiv_gpt/README.md` (the corresponding README).
* Figure 8-13 / Appendix I
    * See the steps for replicating Figure 5
* Table 9
    * See steps for replicating Table 8, but run the `individual_vs_aggregate_gpt` script instead
* Table 10
    * See steps for replicating Table 8, but run the `post_arxiv_gpt/individual/annotator_tagging.py` script instead
* Table 11
    * See steps for replicating Table 8, but run the `post_arxiv_gpt/distributional` script instead
* Table 12
    * See steps for replicating Table 8, but run the `post_arxiv_gpt/individual/data_ablation.py` script instead
* Table 13
    * See steps for replicating Table 8, but run the `post_arxiv_gpt/individual/header_ablation.py` script instead
* Extra Utilities
    * Data Cleaning Scripts (already used to create the datasets in the `datasets/cleaned` and `datasets/cleaned_folds` folder, but helpful for cleaning your own datasets)
        * `utilities/check_duplicate_texts.py`
            * Checks to see if there are any duplicate examples in a dataset. If this test comes back with duplicates, the utilities/fix_duplicate_texts.py script should be used.
        * `utilites/fix_duplicate_texts.py`
            * Merges duplicate examples. If there are conflicting labels, either raises an error, or can choose a label at random (which is what we did for our datasets)
        * `utilities/double_dataset.py` (used in `utilities/double_SBIC.bat`)
            * Some datasets (such as SBIC) use 0.5 increments for labels. This script doubles all the values in a dataset so that labels are all integer labels
    * `utilities/compute_disagreement/compute_disagreement.py`
        * Computes disagreement in a dataset; used in `split_based_on_disagreement/main.py`
    * `utilities/prediction_fold_split.py`
        * Splits the dataset into folds used by other scripts. (If you are replicating our paper, these splits are already provided in the datasets folder.)

Since these scripts were run at different points in time with different file structures, you will likely need to ensure that the scripts point to the correct file locations while running. Everything else should likely remain as the default.

As mentioned above, if you encounter any issues, please leave a GitHub issue and email lowma016@umn.edu if the GitHub issue is not addressed.