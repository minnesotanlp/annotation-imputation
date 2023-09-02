'''
Imputes a matrix using the multitask model.
'''
import numpy as np
from typing import Tuple, Optional
import json
import torch
from tqdm import tqdm
from advanced_logger import AdvancedLogger, AdvancedLoggerType
from data_loading import load_data, load_one_data, get_tokenizer
from training import logits_to_annotations, train
from model_building import RatingModel
from numpy_json_encoder import NumpyJSONEncoder

# import from kernel_matrix_factorization
import sys
sys.path.append('../kernel_matrix_factorization')
from imputer import NpyImputer, JSONstr
from dataset_formats import NpyAnnotationsFormat, NpyTextsFormat
from one_anno_per_cross_split import npy_to_npy_one_anno_per_cross_split

class MultitaskImputer(NpyImputer):
    def __init__(self, name: str, encoder_model: str, epochs: int=10, train_split: float=0.95, logger: Optional[AdvancedLogger]=None):
        super().__init__(name)

        if logger is None:
            logger = AdvancedLogger(AdvancedLoggerType.NONE, params={AdvancedLogger.SAVE_JSON_KEY: True})

        self.encoder_model: str = encoder_model
        self.epochs: int = epochs
        self.train_split: float = train_split
        self.val_split: float = 1 - train_split
        self.logger: AdvancedLogger = logger

    def impute(self, annotations: Optional[NpyAnnotationsFormat], texts: Optional[NpyTextsFormat]) -> Tuple[NpyAnnotationsFormat, JSONstr]:
        assert annotations is not None, "annotations is None."
        assert texts is not None, "texts is None."

        print("Generating train and val splits...")
        train_annotations, train_texts, val_annotations, val_texts, _test_annotations, _test_texts = npy_to_npy_one_anno_per_cross_split(annotations, texts, train_split=self.train_split, val_split=self.val_split)
        print("Train and val splits generated.")

        print("Loading data...")
        train_loader, val_loader = load_data(train_annotations, train_texts, val_annotations, val_texts, self.encoder_model)
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
        model_type = 'multi'
        model = RatingModel(self.encoder_model, model_type, num_annotators, num_unique_ratings)
        print("RatingModel created.")

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device = torch.device("cuda:1")
            else:
                device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        print("Training model...")
        train(model, train_loader, val_loader, device, model_type, self.epochs, self.logger)
        print("Training complete!")

        ## Prediction
        
        # create the dataset for prediction
        # this will consist of all of the texts - the annotations can be empty/zeroed out
        print("Creating prediction dataset...")
        prediction_texts = texts.copy()
        prediction_annotations = np.zeros_like(annotations)
        prediction_loader = load_one_data(prediction_annotations, prediction_texts, get_tokenizer(self.encoder_model))
        
        # predict
        print("Predicting...")
        model.eval()
        with torch.no_grad():
            all_predictions = torch.tensor([]).to(device)
            for input_ids, attention_mask, token_type_ids, _annotations in tqdm(prediction_loader, desc="Predicting"):
                inputs = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "token_type_ids": token_type_ids.to(device)
                }
                _annotations = _annotations.to(device)

                logits = model(inputs)

                assert logits.shape[0] == len(_annotations), f"Mismatch in batch size between logits and annotations. Got {logits.shape[0]} and {len(_annotations)} respectively."

                all_predictions = torch.cat((all_predictions, logits_to_annotations(logits)), dim=0)

                assert all_predictions.ndim == 2, f"all_predictions should have 2 dimensions. Got {all_predictions.ndim} instead."
        
        print("Predictions complete!")
        assert all_predictions.shape == annotations.shape, f"all_predictions and annotations should have the same shape. Got {all_predictions.shape} and {annotations.shape} respectively."

        # convert the predictions to a numpy array
        all_predictions = all_predictions.cpu().numpy()

        # determine how many of the predictions were incorrect
        # ignore the -1 ratings
        incorrect_predictions = (all_predictions != annotations) & (annotations != -1)
        num_incorrect_predictions = incorrect_predictions.sum()
        possible_predictions = (annotations != -1).sum()
        incorrect_prediction_rate = num_incorrect_predictions / possible_predictions

        print(f"Incorrect prediction rate: {round(incorrect_prediction_rate * 100, 2)}%=({num_incorrect_predictions}/{possible_predictions})")

        json_dict = {}
        if self.logger.json:
            json_dict = self.logger.json

        json_dict["incorrect_prediction_rate"] = incorrect_prediction_rate
        json_dict["num_incorrect_predictions"] = num_incorrect_predictions
        json_dict["num_possible_predictions"] = possible_predictions

        # replace all the incorrect predictions with the original annotations
        all_predictions[incorrect_predictions] = annotations[incorrect_predictions]

        # make sure that all of the annotations are now correct
        assert ((all_predictions != annotations) & (annotations != -1)).sum() == 0, f"There are still incorrect predictions. There are {((all_predictions != annotations) & (annotations != -1)).sum()} of them."

        # make sure none of the annotations are -1
        assert (all_predictions == -1).sum() == 0, f"There are still -1 annotations. There are {(all_predictions == -1).sum()} of them."

        json_str = json.dumps(json_dict, indent=4, cls=NumpyJSONEncoder)
        return all_predictions, json_str
    
if __name__ == "__main__":
    import numpy as np
    logger_params = {"project": "multi_anno", "name": "multitask_imputer0", AdvancedLogger.SAVE_JSON_KEY: True}
    logger = AdvancedLogger(AdvancedLoggerType.WANDB, logger_params)
    # logger = AdvancedLogger(AdvancedLoggerType.NONE, logger_params)
    imputer = MultitaskImputer("multitask_imputer", "bert-base-uncased", epochs=10, logger=logger)
    annotations_file = "../datasets/cleaned/SChem_annotations.npy"
    texts_file = "../datasets/cleaned/SChem_texts.npy"
    annotations = np.load(annotations_file, allow_pickle=True)
    texts = np.load(texts_file, allow_pickle=True)
    imputed_data, extra_data = imputer.impute(annotations, texts)
    
    # save the extra data to a json test file with indenting
    json_dict = json.loads(extra_data)
    with open("multitask_imputer_test.json", "w") as f:
        json.dump(json_dict, f, indent=4)