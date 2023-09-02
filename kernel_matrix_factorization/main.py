from kernel_matrix_factorization_imputer import KernelMatrixFactorizationImputer
import argparse
from matrix_to_npy_format import matrix_to_npy_format
from npy_to_matrix_format import npy_to_matrix_format
from dataset_formats import MatrixDataFrame, NpyAnnotationsFormat, NpyTextsFormat
from remove_fold import remove_fold_both
from imputer import JSONstr
import numpy as np
from make_path import make_path
from compare_mixed_dataframes import compare_mixed_dataframes

def main():
    def auto_or_float(value):
        if value == 'auto':
            return value
        try:
            float_value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not a valid float or 'auto'")
        return float_value

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=False,
                        help='dataset to impute (currently not used)')
    parser.add_argument('--name',
                        type=str,
                        default="KernelMatrixFactorizationImputer",
                        required=False,
                        help='name of the imputer')
    parser.add_argument('--n_factors',
                        type=int,
                        nargs='+',
                        default=[32, 16, 8, 4, 2, 1],
                        required=False,
                        help='list of number of factors to use in the grid search')
    parser.add_argument('--n_epochs',
                        type=int,
                        nargs='+',
                        default=[256, 128, 64, 32, 16, 8, 4, 2, 1],
                        required=False,
                        help='list of number of epochs to use in the grid search')
    parser.add_argument('--kernels',
                        type=str,
                        nargs='+',
                        choices=["linear", "rbf", "sigmoid"],
                        default=["linear", "rbf", "sigmoid"],
                        required=False,
                        help='list of kernels to use in the grid search')
    parser.add_argument('--gammas',
                        type=auto_or_float,
                        nargs='+',
                        default=["auto"],
                        required=False,
                        help="A list of potential gamma values. "
                         "Each value can be either 'auto' or a float. "
                         "If 'auto' is used for any value, it will be set to 1/n_factors. "
                         "If a number is used, it will be used as the value of gamma.")
    parser.add_argument('--regs',
                        type=float,
                        nargs='+',
                        default=[0.01],
                        required=False,
                        help='list of regularization values to use in the grid search')
    parser.add_argument('--lrs',
                        type=float,
                        nargs='+',
                        default=[0.01, 0.001, 0.0001],
                        required=False,
                        help='list of learning rates to use in the grid search')
    parser.add_argument('--init_means',
                        type=float,
                        nargs='+',
                        default=[0],
                        required=False,
                        help='list of initial mean values to use in the grid search')
    parser.add_argument('--init_sds',
                        type=float,
                        nargs='+',
                        default=[0.1],
                        required=False,
                        help='list of initial standard deviation values to use in the grid search')
    parser.add_argument('--seeds',
                        type=int,
                        nargs='+',
                        default=[85],
                        required=False,
                        help='list of random seeds to use in the grid search')
    parser.add_argument('--bound_ratings',
                        type=bool,
                        default=True,
                        required=False,
                        help='whether to bound ratings to the range of the given data')
    parser.add_argument('--fold',
                        type=int,
                        default=-1,
                        required=False,
                        help='fold of data to completely ignore (and not impute). -1 means no fold is ignored (impute the entire given dataset).')
    parser.add_argument('--n_folds',
                        type=int,
                        default=5,
                        required=False,
                        help='number of folds to use for determining how to remove the given fold of data')
    parser.add_argument('--train_frac',
                        type=float,
                        default=0.9,
                        required=False,
                        help='fraction of data to use for training')
    parser.add_argument('--val_frac',
                        type=float,
                        default=0.05,
                        required=False,
                        help='fraction of data to use for validation')
    parser.add_argument('--test_frac',
                        type=float,
                        default=0.05,
                        required=False,
                        help='fraction of data to use for testing')
    parser.add_argument('--log_file',
                        type=str,
                        default='matrix_factorization_imputer_log.txt',
                        help='file to log the imputer information to')
    parser.add_argument('--save_imputations',
                        type=bool,
                        default=False,
                        required=False,
                        help='whether or not to save every imputation in the grid search to the json file (can be very memory intensive)')
    parser.add_argument('--allow_duplicates',
                        action='store_const',
                        const=True,
                        default=False,
                        required=False,
                        help='whether to allow duplicate sentences in the dataset')
    parser.add_argument('--input_npy_annotations_file',
                        type=str,
                        required=True,
                        help='path to the npy annotations file')
    parser.add_argument('--input_npy_texts_file',
                        type=str,
                        required=True,
                        help="path to the npy texts file (This isn't really used, but is required for formatting purposes. It's fine if each sentence in the file is just a string of the index of the sentence. However, if there are duplicate sentences, it will cause an issue.)")
    parser.add_argument('--output_npy_annotations_file',
                        type=str,
                        required=True,
                        help='path to the output npy annotations file (will be created if it does not exist)')
    parser.add_argument('--output_json_file',
                        type=str,
                        required=True,
                        help='path to the output json file (will be created if it does not exist)')                    
    
    args = parser.parse_args()

    npy_annotations_file = args.input_npy_annotations_file
    npy_texts_file = args.input_npy_texts_file
    
    with open(npy_annotations_file, 'rb') as f:
        annotations: NpyAnnotationsFormat = np.load(f, allow_pickle=True)

    with open(npy_texts_file, 'rb') as f:
        texts: NpyTextsFormat = np.load(f, allow_pickle=True)

    if args.fold != -1:
        annotations, texts = remove_fold_both(annotations, texts, args.fold, args.n_folds)

    print("Converting from npy format to matrix format...")
    matrix_df = npy_to_matrix_format(annotations, texts, allow_duplicates=args.allow_duplicates)
    print("Conversion complete")

    make_path(args.log_file)

    imputer = KernelMatrixFactorizationImputer(
        name=args.name,
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        kernels=args.kernels,
        gammas=args.gammas,
        regs=args.regs,
        lrs=args.lrs,
        init_means=args.init_means,
        init_sds=args.init_sds,
        seeds=args.seeds,
        bound_ratings=True,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        log_file=args.log_file,
        save_imputations=args.save_imputations
    )

    imputed_matrix_df: MatrixDataFrame
    json_str: JSONstr
    print("Imputing...")
    imputed_matrix_df, json_str = imputer.impute(matrix_df.copy())
    print("Imputation complete")

    print("Checking to make sure output is valid...")
    assert matrix_df.shape == imputed_matrix_df.shape, f"The shape of the imputed matrix is not the same as the shape of the original matrix. Original shape: {matrix_df.shape}. Imputed shape: {imputed_matrix_df.shape}."

    # check to make sure that the unique values in the imputed matrix's first column are the same as the unique values in the original matrix's first column
    assert set(matrix_df.iloc[:, 0]) == set(imputed_matrix_df.iloc[:, 0]), f"The unique values in the first column of the imputed matrix are not the same as the unique values in the first column of the original matrix. Those in the original matrix but not in imputed: {set(matrix_df.iloc[:, 0]) - set(imputed_matrix_df.iloc[:, 0])}. Those in the imputed matrix but not in the original: {set(imputed_matrix_df.iloc[:, 0]) - set(matrix_df.iloc[:, 0])}"

    # check to make sure that the imputed matrix's first column is the same as the original matrix's first column
    assert np.all(matrix_df.iloc[:, 0] == imputed_matrix_df.iloc[:, 0]), f"The first column of the imputed matrix is not exactly the same as the first column of the original matrix. Compare\n{matrix_df.iloc[:, 0]}\nand\n{imputed_matrix_df.iloc[:, 0]}"

    # # assert that every value in the matrix_df is still in the imputed_matrix_df
    # # Get boolean masks for non-null values
    mask_matrix = ~matrix_df.isnull()
    masked_matrix_df = matrix_df[mask_matrix]
    masked_imputed_matrix_df = imputed_matrix_df[mask_matrix]
    assert list(masked_matrix_df.columns) == list(masked_imputed_matrix_df.columns), f"The columns of the imputed matrix are not the same as the columns of the original matrix. Differences: {[(original, imputed) for original, imputed in zip(masked_matrix_df.columns, masked_imputed_matrix_df.columns) if original != imputed]}"
    # assert np.allclose(masked_matrix_df, masked_imputed_matrix_df, equal_nan=True), f"Values in the original matrix were not maintained throughout imputation. Compare\n{matrix_df[mask_matrix]}\nand\n{imputed_matrix_df[mask_matrix]}"
    assert compare_mixed_dataframes(masked_matrix_df, masked_imputed_matrix_df), f"Values in the original matrix were not maintained throughout imputation. Compare\n{matrix_df[mask_matrix]}\nand\n{imputed_matrix_df[mask_matrix]}"

    # # if the above assertion triggers, then comment it out and run this one instead. It's slower, but it will tell you an exact location of where the values differ
    # # assert that every value in the matrix_df is still in the imputed_matrix_df
    # from tqdm import tqdm
    # for i in tqdm(range(matrix_df.shape[0]), leave=False, desc="Checking rows"):
    #     for j in tqdm(range(matrix_df.shape[1]), leave=False, desc=f"Checking row {i}"):
    #         if not pd.isnull(matrix_df.iloc[i, j]):
    #             assert matrix_df.iloc[i, j] == imputed_matrix_df.iloc[i, j], f"Value at ({i}, {j}) ({matrix_df.iloc[i, 0], matrix_df.columns[j]}) ({imputed_matrix_df.iloc[i, 0], imputed_matrix_df.columns[j]}) did not maintain its original value. It was {matrix_df.iloc[i, j]} but is now {imputed_matrix_df.iloc[i, j]}. Did your imputer change the order of the rows or columns?"

    print("Converting back to npy format...")
    imputed_annotations, imputed_texts = matrix_to_npy_format(imputed_matrix_df)
    print("Conversion complete")

    # check that the imputed texts are the same as the original texts
    assert np.all(imputed_texts == texts), f"The texts in the imputed matrix are not the same as the texts in the original matrix. Differences: {[(original, imputed) for original, imputed in zip(texts, imputed_texts) if original != imputed]}. Did your imputer change the order of the rows?"

    output_npy_annotations_file = args.output_npy_annotations_file
    output_json_file = args.output_json_file
    
    make_path(output_npy_annotations_file)
    make_path(output_json_file)

    with open(output_npy_annotations_file, 'wb') as f:
        np.save(f, imputed_annotations)
    print(f"Saved imputed annotations to {output_npy_annotations_file}")

    with open(output_json_file, 'w') as f:
        f.write(json_str)
    print(f"Saved imputation data to {output_json_file}")

    print("Imputation complete!")

if __name__ == "__main__":
    main()