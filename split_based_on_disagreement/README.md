# split_based_on_disagreement/main.py
The goal of the scripts in this folder is to compute whether the Multitask model's predictions on imputed data tend to do better or worse depending on the disagreement in the original (or imputed) dataset.

There are two modes for this script: regular executable, and full run.

## Regular Executable
In this mode, you should run `python3 main.py --input_orig_data <your_filename_here> --trained_json <your_filename_here> --output_json <your_filename_here> --grouping_method var --highest`. You can find information about the parameters with `--help`. However, it's important to note that you can define what you mean by "low", "medium", and "high" disagreement. For our results (using `var` as the `--grouping_method`), we just essentially try to put as even an amount in each category as possible. (So even if there's very little disagreement in a dataset, about 1/3 of the examples will still be labeled "high disagreement".) The details of how this is done are explained in Appendix J.

The script should output a json file to wherever the `output_json` file was set to in the following format:
```
{
    CHOSEN_EPOCH_KEY: best_epoch,
    HIGHEST_KEY: args.highest,
    LOW_DISAGREEMENT_KEY: {
        PREDICTIONS_KEY: <list of predictions made on low disagreement examples>,
        TRUE_LABELS_KEY: <list of true labels for those low disagreement examples>,
        N_EXAMPLES_KEY: <how many text examples/rows in the original dataset were low disagreement>, 
        CLASSIFICATION_REPORT_KEY: <sklearn.metrics.classification_report on low disagreement predictions>
    },
    MEDIUM_DISAGREEMENT_KEY: {
        PREDICTIONS_KEY: <list of predictions made on medium disagreement examples>,
        TRUE_LABELS_KEY: <list of true labels for those medium disagreement examples>,
        N_EXAMPLES_KEY: <how many text examples/rows in the original dataset were medium disagreement>, 
        CLASSIFICATION_REPORT_KEY: <sklearn.metrics.classification_report on medium disagreement predictions>
    },
    HIGH_DISAGREEMENT_KEY: {
        PREDICTIONS_KEY: <list of predictions made on high disagreement examples>,
        TRUE_LABELS_KEY: <list of true labels for those high disagreement examples>,
        N_EXAMPLES_KEY: <how many text examples/rows in the original dataset were high disagreement>, 
        CLASSIFICATION_REPORT_KEY: <sklearn.metrics.classification_report on high disagreement predictions>
    }
}
# note that N_EXAMPLES is not equivalent to the length of PREDICTIONS, as multiple predictions can be made per example
```

This data can then be used to replicate Table 7.

## Full Run
To activate Full Run mode (called `quick_test` in the code, despite it not being quick and not being a test), merely run `python3 main.py` without any parameters.

This will attempt to compute results on the SChem, GHC, Politeness, and Sentiment datasets. (If you're running into issues, reading the code to understand how that is done may be informative. If it's still not clear how to use this, please leave a GitHub issue.)

It will then also compute the exact values used in Table 7, and output them to a file set by the constant `overall_outputs_path` (the code can be edited to change this location). (It will also compute all of the values to make a table for accuracy, in addition to F1 score. However, Table 7 only reports the F1 scores.)

This is what was used to actually produce Table 7.

(Also, `graph.py` should be functional and create a graph of the data, but it's not used in the paper and hasn't been tested under the Docker image, so no promises.)