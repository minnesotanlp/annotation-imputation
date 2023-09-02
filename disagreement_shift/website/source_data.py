'''
Grab data from the dataset folder and turn it into the right format for the website.
'''
import os

main_dataset_folder = "../../datasets"
ncf_datasets_folder = f"{main_dataset_folder}/final_imputation_results/ncf_imputation_results"
kernel_datasets_folder = f"{main_dataset_folder}/final_imputation_results/kernel_imputation_results"

ncf_imputation = f"{ncf_datasets_folder}/{{dataset_name}}_-1_ncf_imputation_-1.npy"
kernel_imputation = f"{kernel_datasets_folder}/Fold_-1/{{dataset_name}}_annotations_kernel_preds.npy"
ncf_distribution = f"{main_dataset_folder}/final_imputation_results/distribution_analysis/ncf/{{dataset_name}}_imputed_distribution.npy"
ncf_kl = f"{main_dataset_folder}/final_imputation_results/distribution_analysis/ncf/{{dataset_name}}_kl.npy"
kernel_distribution = f"{main_dataset_folder}/final_imputation_results/distribution_analysis/kernel/{{dataset_name}}_imputed_distribution.npy"
kernel_kl = f"{main_dataset_folder}/final_imputation_results/distribution_analysis/kernel/{{dataset_name}}_kl.npy"
orig_annotations = f"{main_dataset_folder}/cleaned/{{dataset_name}}_annotations.npy"
orig_distribution = f"{main_dataset_folder}/final_imputation_results/distribution_analysis/original/{{dataset_name}}_orig_distribution.npy"
orig_texts = f"{main_dataset_folder}/cleaned/{{dataset_name}}_texts.npy"

dataset_title = "Politeness"
dataset_name = "politeness"

# create the folder
website_folder = f"data/{dataset_title}"
os.makedirs(website_folder, exist_ok=True)

# add the ncf and kernel imputations
with open(ncf_imputation.format(dataset_name=dataset_name), "rb") as f:
    with open(f"{website_folder}/imputed_ncf_annotations.npy", "wb") as f2:
        f2.write(f.read())

with open(kernel_imputation.format(dataset_name=dataset_name), "rb") as f:
    with open(f"{website_folder}/imputed_kernel_annotations.npy", "wb") as f2:
        f2.write(f.read())

# add the ncf and kernel distributions and KL
with open(ncf_distribution.format(dataset_name=dataset_name), "rb") as f:
    with open(f"{website_folder}/imputed_ncf_distribution.npy", "wb") as f2:
        f2.write(f.read())

with open(ncf_kl.format(dataset_name=dataset_name), "rb") as f:
    with open(f"{website_folder}/ncf_kl.npy", "wb") as f2:
        f2.write(f.read())

with open(kernel_distribution.format(dataset_name=dataset_name), "rb") as f:
    with open(f"{website_folder}/imputed_kernel_distribution.npy", "wb") as f2:
        f2.write(f.read())

with open(kernel_kl.format(dataset_name=dataset_name), "rb") as f:
    with open(f"{website_folder}/kernel_kl.npy", "wb") as f2:
        f2.write(f.read())

# add the original annotations, distribution, and texts
with open(orig_annotations.format(dataset_name=dataset_name), "rb") as f:
    with open(f"{website_folder}/orig_annotations.npy", "wb") as f2:
        f2.write(f.read())

with open(orig_distribution.format(dataset_name=dataset_name), "rb") as f:
    with open(f"{website_folder}/orig_distribution.npy", "wb") as f2:
        f2.write(f.read())

with open(orig_texts.format(dataset_name=dataset_name), "rb") as f:
    with open(f"{website_folder}/texts.npy", "wb") as f2:
        f2.write(f.read())

print("Done!")