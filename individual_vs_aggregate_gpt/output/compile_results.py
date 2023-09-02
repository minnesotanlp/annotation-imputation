import os
import json

all_results = {}
# iterate through each folder in the current directory
for folder in os.listdir(os.getcwd()):
    # check to make sure "Seed" is in the folder name
    if "Seed" in folder:
        folder_name = None
        replacements = {"K_-1": "Maj_Voted", "K_None": "Individual", "K_distributional": "Original_Distribution_Leaked", "K_imputed_distributional": "Imputed_Distribution_Leaked", "K_original_distributional_correct": "Original_Distribution_Correct", "K_imputed_distributional_correct": "Imputed_Distribution_Correct"}
        found_replacement = False
        for key in replacements:
            if folder.endswith(key):
                found_replacement = True
                folder_name = folder[:-len(key)] + replacements[key]
        if not found_replacement:
            raise ValueError(f"Invalid folder name: {folder}")
        # open the statistics.json file
        with open(folder + "/statistics.json") as f:
            # load the json file
            data = json.load(f)
            all_results[folder_name] = data

# write the results to a file
with open("all_results.json", "w") as f:
    json.dump(all_results, f, indent=4)