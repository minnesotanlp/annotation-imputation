import torch
from arg_parsing import parse_arguments
from data_loading import load_data_from_paths
from model_building import RatingModel
from training import train
from advanced_logger import AdvancedLogger, AdvancedLoggerType
import numpy as np
import json
import os
from numpy_json_encoder import NumpyJSONEncoder

def make_path(filename: str):
    '''Makes all the required directories for a file path to be valid.'''
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

def main():
    args = parse_arguments()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading data...")
    train_loader, val_loader = load_data_from_paths(args.train_annotations, args.train_texts, args.val_annotations, args.val_texts, args.encoder_model)
    print("Data loaded.")

    num_annotators = train_loader.dataset.annotations.shape[1]
    unique_ratings = torch.unique(train_loader.dataset.annotations)
    # remove the -1 rating explicitly
    unique_ratings = unique_ratings[unique_ratings != -1]
    num_unique_ratings = unique_ratings.shape[0]

    # make sure that 0 is the lowest rating and num_unique_ratings - 1 is the highest rating
    assert unique_ratings[0] == 0, f"Lowest rating is not 0: {unique_ratings[0]}."
    assert unique_ratings[-1] == num_unique_ratings - 1, f"Highest rating is not {num_unique_ratings - 1}: {unique_ratings[-1]}."

    print("Creating RatingModel...")
    model = RatingModel(args.encoder_model, args.model_type, num_annotators, num_unique_ratings)
    print("RatingModel created.")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1") # using second GPU
    # device = torch.device("cpu") # debugging

    logger_params = {"project": args.wandb_project, "name": args.wandb_run_name, AdvancedLogger.SAVE_JSON_KEY: True}
    if args.logger_type == "tensorboard":
        logger_type = AdvancedLoggerType.TENSORBOARD
    elif args.logger_type == "wandb":
        logger_type = AdvancedLoggerType.WANDB
    else:
        logger_type = AdvancedLoggerType.NONE

    logger = AdvancedLogger(logger_type, logger_params)

    print("Training model...")
    train(model, train_loader, val_loader, device, args.model_type, args.epochs, logger)
    print("Training complete!")

    if args.save_logs:
        save_location = args.save_logs_where
        make_path(save_location)
        if logger.json is None:
            print("No logs to save.")
        else:
            with open(save_location, "w") as f:
                json.dump(logger.json, f, indent=4, cls=NumpyJSONEncoder)

if __name__ == "__main__":
    main()