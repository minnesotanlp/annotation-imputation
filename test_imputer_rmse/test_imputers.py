'''
Tests imputers on a dataset.
'''

if __name__ == "__main__":
    import argparse
    import pandas as pd
    from dataclasses import replace
    from tqdm import tqdm
    import os
    import json
    from evaluate_imputers import get_rmse_score, get_classification_report
    from math import isnan

    # imports from `kernel_matrix_factorization`
    import sys
    sys.path.append("../kernel_matrix_factorization")
    from make_path import make_path
    from save_df import save_df
    from dataset_columns import SENTENCE_ID_COL, SENTENCE_COL
    from dataset_format_conversions import matrix_to_factor_form, matrix_to_npy_format
    from kernel_matrix_factorization_imputer import KernelMatrixFactorizationImputer
    from imputer import NpyImputerWrapper, NpyImputer, Imputer

    # imports from `ncf_matrix_factorization`
    sys.path.append("../ncf_matrix_factorization")
    from ncf_matrix_factorization_imputer import NCFMatrixFactorizationImputer

    # imports from `median_imputer`
    sys.path.append("../median_imputer")
    from median_imputer import MedianImputer, Direction

    # imports from multitask
    sys.path.append("../multitask")
    from numpy_json_encoder import NumpyJSONEncoder
    from multitask_imputer import MultitaskImputer, AdvancedLogger, AdvancedLoggerType

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Location of the train annotations CSV file (Matrix Dataset format).")
    parser.add_argument("--test", type=str, required=True, help="Location of the test annotations CSV file (Matrix Dataset format).")
    parser.add_argument("--output", type=str, required=True, help="Location to save the results.")
    parser.add_argument("--data", type=str, default=None, help="Folder to store run data in. If not specified, no data will be saved.")
    parser.add_argument("--python_command", type=str, default="python", help="Command to use to run python scripts.")
    parser.add_argument("--ncf_batch_size", type=int, default=256, help="Batch size to use for NCF imputer.")
    parser.add_argument("--ncf_epochs", type=int, default=20, help="Number of epochs to use for NCF imputer.")
    parser.add_argument("--ncf_main_ours_search", type=str, default=None, help="Location of the NCF main_ours_search.py script.")
    parser.add_argument("--multitask_epochs", type=int, default=10, help="Number of epochs to use for multitask imputer.")
    parser.add_argument("--multitask_encoder_model", type=str, default='bert-base-uncased', help="Encoder model to use for multitask imputer.")
    parser.add_argument("--multitask_train_split", type=float, default=0.95, help="Train split to use for multitask imputer.")
    parser.add_argument("--multitask_logger_type", type=str, default='none', help="Logger type to use for multitask imputer.")
    parser.add_argument("--multitask_logger_params", type=str, default='{}', help="Logger parameters to use for multitask imputer. In the form of a JSON string.")
    args = parser.parse_args()

    row_median_imputer = MedianImputer("RowMedianImputer", Direction.SENTENCE, error_on_empty=False)
    col_median_imputer = MedianImputer("ColumnMedianImputer", Direction.USER, error_on_empty=False)
    global_median_imputer = MedianImputer("GlobalMedianImputer", Direction.GLOBAL, error_on_empty=False)

    ncf_imputer = NCFMatrixFactorizationImputer("NCFMatrixFactorizationImputer", args.python_command, args.ncf_main_ours_search, args.ncf_batch_size, args.ncf_epochs)

    try:
        multitask_logger_params = json.loads(args.multitask_logger_params)
    except json.decoder.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON string for multitask_logger_params: {args.multitask_logger_params}") from e
    multitask_logger_type: AdvancedLoggerType = AdvancedLoggerType.from_str(args.multitask_logger_type)
    multitask_logger = AdvancedLogger(multitask_logger_type, multitask_logger_params)
    npy_multitask_imputer: NpyImputer = MultitaskImputer(name="MultitaskImputer",
                                         encoder_model=args.multitask_encoder_model,
                                         epochs=args.multitask_epochs,
                                         train_split=args.multitask_train_split,
                                         logger=multitask_logger)
    multitask_imputer: Imputer = NpyImputerWrapper(npy_multitask_imputer)
    
    n_factors = [64, 32, 16, 8, 4, 2, 1]
    # n_factors = [4, 2, 1]
    n_epochs = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    # n_epochs = [4, 2, 1]
    kernels = ["linear", "rbf", "sigmoid"]
    gammas = ["auto"]
    regs = [0.01]
    lrs = [0.01, 0.001, 0.0001]
    init_means = [0]
    init_sds = [0.1]
    # Each list of seeds corresponds to a imputer that will do a hyperparameter search over those seeds
    # seeds = [[30], [26], [46]]
    seeds = [[85]]

    mf_imputers = []
    for seed_index in range(len(seeds)):
        seed = seeds[seed_index]
        mf_imputer = KernelMatrixFactorizationImputer(
            name=f"KernelMatrixFactorizationSeed{seed[0]}" if len(seed) == 1 else f"KernelMatrixFactorizationSeed{seed_index}",
            n_factors=n_factors,
            n_epochs=n_epochs,
            kernels=kernels, gammas=gammas,
            regs=regs,
            lrs=lrs,
            init_means=init_means,
            init_sds=init_sds,
            seeds=seed)
        mf_imputers.append(mf_imputer)

    imputers = [multitask_imputer] + [ncf_imputer] + [row_median_imputer, col_median_imputer] + mf_imputers
    
    json_data = {}
    if args.data is not None:
        make_path(args.data + "/")

    # load the dataset
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Assert that the labels in the train and test sets are all integers, except for the Nones
    # Create a boolean mask where the DataFrame is not None
    assert train_df.iloc[:, 2:].applymap(lambda x: isnan(x) or x==int(x)).all().all()

    # check that the values of the SENTENCE_ID_COL and SENTENCE_COL are the same in the train and test sets
    # I'm no longer sure why this is necessary - it may have something to do with the switching back and forth between matrix and factor form
    assert train_df[SENTENCE_ID_COL].equals(test_df[SENTENCE_ID_COL])
    assert train_df[SENTENCE_COL].equals(test_df[SENTENCE_COL])

    factor_test_df = matrix_to_factor_form(test_df)

    for imputer in tqdm(imputers, desc="Imputers", leave=False):
        imputer_name = imputer.name
        tqdm.write(f"\nRunning {imputer_name}")
        # impute the dataset
        extra_data: str
        imputed_df, extra_data = imputer.impute(train_df)
        if args.data is not None:
            with open(os.path.join(args.data, imputer_name + ".json"), "w") as f:
                json.dump(json.loads(extra_data), f, indent=4)

        # get the rmse score of the imputed dataset
        tqdm.write("Converting to factor form")
        factor_imputed_df = matrix_to_factor_form(imputed_df)

        # tqdm.write("lengths")
        # tqdm.write(str(len(factor_imputed_df)))
        # tqdm.write(str(len(factor_test_df)))

        tqdm.write("Getting RMSE score")
        rmse_score = get_rmse_score(factor_imputed_df, factor_test_df)
        classification_data = get_classification_report(factor_imputed_df, factor_test_df)
        tqdm.write(f"{imputer_name}: {rmse_score}")
        json_data[imputer_name] = {}
        json_data[imputer_name]["rmse"] = rmse_score
        json_data[imputer_name]["classification_report"] = classification_data
        json_data[imputer_name]["imputation"] = matrix_to_npy_format(imputed_df)

        make_path(args.output)
        with open(args.output, "w") as f:
            json.dump(json_data, f, indent=4, cls=NumpyJSONEncoder)