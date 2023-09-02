'''
Graphs the JSON data from the Matrix Factorization imputation
'''

'''
The JSON data looks like this:
{"<Model: factors=1 epochs=1>": {"epochs": 1, "factors": 1, "val_rmse": 0.9932522920136575}, "<Model: factors=1 epochs=500>": {"epochs": 500, "factors": 1, "val_rmse": 0.8559880950336625}, "<Model: factors=1 epochs=1000>": {"epochs": 1000, "factors": 1, "val_rmse": 0.9207047403729596}, "<Model: factors=1 epochs=1500>": {"epochs": 1500, "factors": 1, "val_rmse": 0.9573139436483367}, "<Model: factors=1 epochs=2000>": {"epochs": 2000, "factors": 1, "val_rmse": 1.0048248434615203}, "<Model: factors=2 epochs=1>": {"epochs": 1, "factors": 2, "val_rmse": 0.9934367341953566}, "<Model: factors=2 epochs=500>": {"epochs": 500, "factors": 2, "val_rmse": 0.848775141414065}, "<Model: factors=2 epochs=1000>": {"epochs": 1000, "factors": 2, "val_rmse": 0.9189994778391894}, "<Model: factors=2 epochs=1500>": {"epochs": 1500, "factors": 2, "val_rmse": 0.9584665893721401}, "<Model: factors=2 epochs=2000>": {"epochs": 2000, "factors": 2, "val_rmse": 1.0097942414611616}, "<Model: factors=4 epochs=1>": {"epochs": 1, "factors": 4, "val_rmse": 0.9934496094609725}, "<Model: factors=4 epochs=500>": {"epochs": 500, "factors": 4, "val_rmse": 0.8525391156151045}, "<Model: factors=4 epochs=1000>": {"epochs": 1000, "factors": 4, "val_rmse": 0.8902175768878974}, "<Model: factors=4 epochs=1500>": {"epochs": 1500, "factors": 4, "val_rmse": 0.8988762419466839}, "<Model: factors=4 epochs=2000>": {"epochs": 2000, "factors": 4, "val_rmse": 0.903692283955273}, "<Model: factors=8 epochs=1>": {"epochs": 1, "factors": 8, "val_rmse": 0.9935982463690136}, "<Model: factors=8 epochs=500>": {"epochs": 500, "factors": 8, "val_rmse": 0.8502330579942304}, "<Model: factors=8 epochs=1000>": {"epochs": 1000, "factors": 8, "val_rmse": 0.869112334210938}, "<Model: factors=8 epochs=1500>": {"epochs": 1500, "factors": 8, "val_rmse": 0.8743880604982212}, "<Model: factors=8 epochs=2000>": {"epochs": 2000, "factors": 8, "val_rmse": 0.8683912499576361}, "<Model: factors=16 epochs=1>": {"epochs": 1, "factors": 16, "val_rmse": 0.9932929771984719}, "<Model: factors=16 epochs=500>": {"epochs": 500, "factors": 16, "val_rmse": 0.851814577599326}, "<Model: factors=16 epochs=1000>": {"epochs": 1000, "factors": 16, "val_rmse": 0.8579066792018948}, "<Model: factors=16 epochs=1500>": {"epochs": 1500, "factors": 16, "val_rmse": 0.8643715309013279}, "<Model: factors=16 epochs=2000>": {"epochs": 2000, "factors": 16, "val_rmse": 0.8562348974830091}, "<Model: factors=32 epochs=1>": {"epochs": 1, "factors": 32, "val_rmse": 0.9954449108461113}, "<Model: factors=32 epochs=500>": {"epochs": 500, "factors": 32, "val_rmse": 0.8504502614061936}, "<Model: factors=32 epochs=1000>": {"epochs": 1000, "factors": 32, "val_rmse": 0.8551196573102381}, "<Model: factors=32 epochs=1500>": {"epochs": 1500, "factors": 32, "val_rmse": 0.8491203418935659}, "<Model: factors=32 epochs=2000>": {"epochs": 2000, "factors": 32, "val_rmse": 0.8578695097319823}, "<Model: factors=64 epochs=1>": {"epochs": 1, "factors": 64, "val_rmse": 0.9960467308188473}, "<Model: factors=64 epochs=500>": {"epochs": 500, "factors": 64, "val_rmse": 0.8478527186096392}, "<Model: factors=64 epochs=1000>": {"epochs": 1000, "factors": 64, "val_rmse": 0.8544043403260447}, "<Model: factors=64 epochs=1500>": {"epochs": 1500, "factors": 64, "val_rmse": 0.8566477342104707}, "<Model: factors=64 epochs=2000>": {"epochs": 2000, "factors": 64, "val_rmse": 0.8452671977714559}, "<Model: factors=128 epochs=1>": {"epochs": 1, "factors": 128, "val_rmse": 0.998446132980641}, "<Model: factors=128 epochs=500>": {"epochs": 500, "factors": 128, "val_rmse": 0.8561487874281631}, "<Model: factors=128 epochs=1000>": {"epochs": 1000, "factors": 128, "val_rmse": 0.8575475421247709}, "<Model: factors=128 epochs=1500>": {"epochs": 1500, "factors": 128, "val_rmse": 0.8627125412360939}, "<Model: factors=128 epochs=2000>": {"epochs": 2000, "factors": 128, "val_rmse": 0.8591154070635909}, "<Model: factors=256 epochs=1>": {"epochs": 1, "factors": 256, "val_rmse": 1.0023830930265505}, "<Model: factors=256 epochs=500>": {"epochs": 500, "factors": 256, "val_rmse": 0.8803735396392396}, "<Model: factors=256 epochs=1000>": {"epochs": 1000, "factors": 256, "val_rmse": 0.8659828153744369}, "<Model: factors=256 epochs=1500>": {"epochs": 1500, "factors": 256, "val_rmse": 0.8642901497669891}, "<Model: factors=256 epochs=2000>": {"epochs": 2000, "factors": 256, "val_rmse": 0.8694128774783141}, "<Model: factors=512 epochs=1>": {"epochs": 1, "factors": 512, "val_rmse": 1.0199745603275667}, "<Model: factors=512 epochs=500>": {"epochs": 500, "factors": 512, "val_rmse": 0.9070605206578807}, "<Model: factors=512 epochs=1000>": {"epochs": 1000, "factors": 512, "val_rmse": 0.9002624295173792}, "<Model: factors=512 epochs=1500>": {"epochs": 1500, "factors": 512, "val_rmse": 0.8990632357457093}, "<Model: factors=512 epochs=2000>": {"epochs": 2000, "factors": 512, "val_rmse": 0.8879900995152598}, "best_model": "<Model: factors=64 epochs=2000>"}

The goal is to create a scatterplot of how factors affect rmse and (seperately) how epochs affect rmse.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import json
import os
# add .. to the path
import sys
sys.path.append("..")
from dataset_identifier import DatasetIdentifier, Format, Split, ImputationLevel, Aggregation

EPOCHS_KEY = "lr"
FACTORS_KEY = "factors"
RMSE_KEY = "val_rmse"
BEST_MODEL_KEY = "best_model"

# parameters
dataset_names = ['SBIC', 'Hatespeech', 'Sentiment', 'SChem5Labels', 'SChem']
dataset_folder = "../../2-7-2023"
# if None, save at the same location as the json file
# only used if save_visualizations is True
# save_folder = "graph_single_mf_json_images"
save_folder = None
show_visualizations = True
save_visualizations = True

for dataset_name in dataset_names:
    dataset_identifier = DatasetIdentifier(main_folder = dataset_folder, dataset_name=dataset_name, format=Format.MATRIX, split=Split.TRAIN, imputation_level=ImputationLevel.IMPUTED, imputation_method="MatrixFactorization")
    json_location = dataset_identifier.location_with_file("json_data.json")
    json_folder = os.path.dirname(json_location)
    if save_folder is not None:
        # make the folder if it doesn't exist
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        base_folder = save_folder
    else:
        base_folder = json_folder

    with open(json_location, "r") as f:
        data = json.load(f)

    # remove the "best model" key because it's not a data point
    data.pop(BEST_MODEL_KEY)
    # we can discard the initial keys because they hold the same info as a data in them
    data = [values for values in data.values()]
    # print(data)

    # gather the data
    epochs = [data_point[EPOCHS_KEY] for data_point in data]
    factors = [data_point[FACTORS_KEY] for data_point in data]
    rmse = [data_point[RMSE_KEY] for data_point in data]

    # group data by epochs
    data_by_epoch = {}
    for i, epoch in enumerate(epochs):
        if epoch not in data_by_epoch:
            data_by_epoch[epoch] = []
        data_by_epoch[epoch].append((factors[i], rmse[i]))

    # group data by factors
    data_by_factor = {}
    for i, factor in enumerate(factors):
        if factor not in data_by_factor:
            data_by_factor[factor] = []
        data_by_factor[factor].append((epochs[i], rmse[i]))

    # setup markers
    markers = list(matplotlib.markers.MarkerStyle.markers.keys())
    # remove ["None", "none", " ", ""]
    markers = [marker for marker in markers if marker not in ["None", "none", " ", ""]]
    # define the ones I want to use for sure
    default_ordering = ["o", "s", "p", "^", "x", "d", "v", "<", ">", "h", "+"]
    markers = default_ordering + [marker for marker in markers if marker not in default_ordering]

    # plot factors vs rmse for each epoch
    for i, (epoch, points) in enumerate(data_by_epoch.items()):
        x, y = zip(*points)
        plt.scatter(x, y, marker=markers[i % len(markers)], label=f'Epoch {epoch}')
        plt.legend()
    plt.xlabel("Factors")
    plt.ylabel("RMSE")
    plt.title(f"{dataset_name}: Factors vs RMSE (by Epoch)")
    if save_visualizations:
        save_location = os.path.join(base_folder, f"{dataset_name}_factors_vs_rmse_by_epoch.png")
        plt.savefig(save_location)
        print(f"Saved to {save_location}")
    if show_visualizations:
        plt.show()
    else:
        plt.clf()


    # plot epochs vs rmse for each factor
    for i, (factor, points) in enumerate(data_by_factor.items()):
        x, y = zip(*points)
        plt.scatter(x, y, marker=markers[i % len(markers)], label=f'Factor {factor}')
        plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("RMSE")
    plt.title(f"{dataset_name}: Epochs vs RMSE (by Factor)")
    if save_visualizations:
        save_location = os.path.join(base_folder, f"{dataset_name}_epochs_vs_rmse_by_factor.png")
        plt.savefig(save_location)
        print(f"Saved to {save_location}")
    if show_visualizations:
        plt.show()
    else:
        plt.clf()


    # 3D plot of epochs, factors, and rmse (not saved)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(epochs, factors, rmse)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Factors")
    ax.set_zlabel("RMSE")
    plt.title(f"{dataset_name}: Epochs, Factors, and RMSE")
    if show_visualizations:
        plt.show()
    else:
        plt.clf()