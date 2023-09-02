# Convert a numpy file to a csv file
import pandas as pd
import numpy as np

# datasets = ["SChem", "SChem5Labels", "politeness", "ghc", "Fixed2xSBIC", "Sentiment"]
datasets = ["SChem", "ghc"]

missing_is_empty = False
for dataset in datasets:
    print(f"Working on {dataset}...")
    annotations = np.load(f"../datasets/ncf_imputation_results/{dataset}_0_ncf_imputation_0.npy")
    if missing_is_empty:
        annotations = np.where(annotations == -1, np.nan, annotations)

    # convert to dataframe
    annotations_df = pd.DataFrame(annotations)
    annotations_df.to_csv(f"../datasets/csv/imputed/ncf/{dataset}_{'' if missing_is_empty else '-1_'}annotations.csv", index=False, header=False)