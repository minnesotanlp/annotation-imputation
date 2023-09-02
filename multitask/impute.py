import argparse
import numpy as np
from multitask_imputer import MultitaskImputer
from advanced_logger import AdvancedLogger, AdvancedLoggerType
import json
from tqdm import tqdm
from numpy_json_encoder import NumpyJSONEncoder

# import from ../kernel_matrix_factorization
import sys
sys.path.append('../kernel_matrix_factorization')
from remove_fold import remove_fold
from make_path import make_path

parser = argparse.ArgumentParser()
parser.add_argument("--annotations", type=str, required=True, help="Path to the annotations file.")
parser.add_argument("--texts", type=str, required=True, help="Path to the texts file.")
parser.add_argument("--output_annotations", type=str, required=True, help="Path for imputed output annotations file.")
parser.add_argument("--output_texts", type=str, required=True, help="Path for imputed output texts file.")
parser.add_argument("--output_json", type=str, required=True, help="Path for imputed output json file.")
parser.add_argument("--fold", type=int, default=-1, help="The fold to remove. If -1, then no fold will be removed.")
parser.add_argument("--encoder_model", type=str, default="bert-base-uncased", help="The encoder model to use.")
parser.add_argument("--epochs", type=int, default=5, help="The number of epochs to train for.")
parser.add_argument("--train_split", type=float, default=1, help="The proportion of the data to use for training.")
parser.add_argument("--n_folds", type=int, default=5, help="The number of folds in the data.")
parser.add_argument("--logger_type", type=str, choices=[poss.value for poss in AdvancedLoggerType], default=AdvancedLoggerType.WANDB.value, help="The type of logger to use.")
parser.add_argument("--logger_params", type=str, default="{}", help="The parameters to pass to the logger as a json string.")

args = parser.parse_args()

# make the locations for the output files
make_path(args.output_annotations)
make_path(args.output_texts)
make_path(args.output_json)

annotations = np.load(args.annotations, allow_pickle=True)
texts = np.load(args.texts, allow_pickle=True)

if args.fold != -1:
    annotations = remove_fold(annotations, fold=args.fold, n_folds=args.n_folds)
    texts = remove_fold(texts, fold=args.fold, n_folds=args.n_folds)

# create logger
logger_type = AdvancedLoggerType.from_str(args.logger_type)
try:
    logger_params = json.loads(args.logger_params)
except json.decoder.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON string for logger_params: {args.logger_params}") from e
logger = AdvancedLogger(logger_type, logger_params)

# create a multitask imputer
imputer = MultitaskImputer(name="Imputer", encoder_model=args.encoder_model, epochs=args.epochs, train_split=args.train_split, logger=logger)

# impute
imputed_annotations, json_data_str = imputer.impute(annotations, texts)

# save the imputed annotations
np.save(args.output_annotations, imputed_annotations)
np.save(args.output_texts, texts)

# save the json data
with open(args.output_json, "w") as f:
    json.dump(json.loads(json_data_str), f, indent=4, cls=NumpyJSONEncoder)