# Disagreement Site
This is a website that shows the disagreement in three different datasets: an original dataset, and two different imputed datasets (currenly tailored for `kernel` and `ncf`).

Put your data into the `website/data` folder (you may have to create this folder). You should have 9 files with these names inside `website/data/<dataset_name>`:
1. `imputed_kernel_annotations.npy` (the entire imputed dataset from kernel imputation: fold -1)
2. `imputed_kernel_distribution.npy` (distribution from the `label_distribution` script)
3. `imputed_ncf_annotations.npy` (the entire imputed dataset from kernel imputation: fold -1)
4. `imputed_ncf_distribution.npy` (distribution from the `label_distribution` script)
5. `kernel_kl.npy` (kl divergences from the `label_distribution` script)
6. `ncf_kl.npy` (kl divergences from the `label_distribution` script)
7. `orig_annnotations.npy` (original annotations file)
8. `orig_distribution.npy` (distribution from the `label_distribution` script)
9. `texts.npy` (original text file)

(If you just want to demo the site, download [this folder of data](https://drive.google.com/drive/folders/1MuhSsBuC2__Cmr0w9mOfF8GhNFSyQSA8?usp=share_link) from our team datasets folder and place it at `website/data`.)

Then, set the `dataset_name` in `app.py` to the name of the folder you're loading the data from (which should be the dataset name as well).

Finally, run with `python app.py`. Visit the url it displays when run. If you have questions, create a GitHub Issue or email lowma016@umn.edu

_This script was created with assistance from GPT-4_