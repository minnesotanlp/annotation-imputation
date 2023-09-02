import numpy as np

dataset_location = "../datasets/cleaned/ghc_annotations.npy"
# dataset_location = "../datasets/cleaned/SBIC_annotations.npy"
# dataset_location = "../datasets/cleaned/SChem_annotations.npy"
# dataset_location = "../datasets/cleaned/politeness_annotations.npy"

# dataset_location = "../datasets/final_imputation_results/kernel_imputation_results/Fold_-1/politeness_annotations_kernel_preds.npy"
# dataset_location = "../datasets/final_imputation_results/original/politeness_annotations.npy"
# dataset_location = "../datasets/final_imputation_results/original/SChem_annotations.npy"

# dataset_location = "../datasets/cleaned_folds/SChem/SChem_train_0_annotations.npy"
# dataset_location = "../datasets/cleaned_folds/SChem/SChem_train_0_texts.npy"
# dataset_location = "../datasets/cleaned_folds/SChem/SChem_0_ncf_imputation_0.npy"
# dataset_location = "../datasets/cleaned_folds/SChem/SChem_val_0_annotations.npy"

# dataset_location = "../datasets/ghc_texts.npy"
# dataset_location = "../datasets/cleaned/ghc_texts.npy"
# dataset_location = "../datasets/disagreement_datasets/kernel/SChem_annotations.npy"
# dataset_location = "../datasets/main_ours_output/ghc_preds.npy"
# dataset_location = "../datasets/main_ours_output/ghc_annotations.npy"
# dataset_location = "../datasets/disagreement_datasets/clean-ncf/distribution/ghc_kl.npy"
# dataset_location = "../datasets/cleaned/embeddings/SChem_embeddings.npy"
# dataset_location = "../datasets/cleaned/SChem_texts.npy"

dataset = np.load(dataset_location, allow_pickle=True)
print(f"Location: {dataset_location}")
print(dataset[:min(10, len(dataset))])
print(dataset.shape)