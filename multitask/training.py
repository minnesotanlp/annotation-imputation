import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from advanced_logger import AdvancedLogger, AdvancedLoggerType
from sklearn.metrics import classification_report, confusion_matrix
from scipy import stats
from tqdm import tqdm
import numpy as np

CLASSIFICATION_REPORT_KEY = "cr"
CONFUSION_MATRIX_KEY = "cm"
INDIVIDUAL_KEY = "individual"
AGGREGATE_KEY = "aggregate"
PREDICTIONS_KEY = "predictions"
TRUE_LABELS_KEY = "true_labels"

def multi_task_loss(y_pred, y_true):
    '''
    Returns the total cross_entropy loss summed over each annotator
    y_pred: (batch_size, num_annotators, num_classes)
    y_true: (batch_size, num_annotators)
    '''
    # Calculate the cross_entropy loss for each annotator
    loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1), reduction='none', ignore_index=-1)
    
    # Reshape the loss back to (batch_size, num_annotators)
    loss = loss.view(y_true.size())
    
    # Sum the loss over each annotator
    total_loss = loss.sum(dim=1).mean()
    
    return total_loss

def majority_vote(annotations):
    '''Takes a 2D array of annotations (batch_size, num_annotators) and returns the majority vote (batch_size,).'''
    return stats.mode(annotations, axis=1, nan_policy='omit', keepdims=False)[0]

def logits_to_annotations(logits):
    '''Convert from logits (batch_size, num_annotators, num_classes) to annotations (batch_size, num_annotators)
    Assumes that ratings are within the range 0 to num_classes - 1
    '''
    return torch.argmax(logits, dim=-1)

def missing_to_nan(annotations, predictions, missing_index=-1):
    '''Replaces missing annotations with NaNs. Determines which annotations are missing based on the annotations.'''
    assert annotations.shape == predictions.shape, f"Annotations and predictions must have the same shape. Got {annotations.shape} and {predictions.shape} instead."
    mask = annotations == missing_index
    annotations[mask] = float('nan')
    predictions[mask] = float('nan')
    assert annotations.shape == predictions.shape, f"Annotations and predictions should have the same shape. Got {annotations.shape} and {predictions.shape} instead."
    return annotations, predictions

def model_training_setup(model, train_loader, device, num_epochs):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)

    return model, optimizer, scheduler

def train_one_epoch(model, train_loader, device, model_type, optimizer, scheduler) -> float:
    '''Trains the model on the training set and returns the average training loss across all batches.'''
    model.train()
    train_loss: float = 0.0
    for input_ids, attention_mask, token_type_ids, annotations in tqdm(train_loader, desc="Training batch"):
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "token_type_ids": token_type_ids.to(device)
        }
        assert inputs["input_ids"].ndim == 2, f"Input must have 2 dimensions (batch_size, sequence_length). Got {inputs['input_ids'].ndim} instead. Shape {inputs['input_ids'].shape}"
        annotations = annotations.to(device)

        optimizer.zero_grad()
        logits = model(inputs)

        assert logits.shape[0] == len(annotations), f"Mismatch in batch size between logits and annotations. Got {logits.shape[0]} and {len(annotations)} respectively."

        if model_type == "base":
            raise NotImplementedError
            # majority_voted_annotations = majority_vote(annotations)
            # loss = nn.CrossEntropyLoss()(logits, majority_voted_annotations.squeeze())
        elif model_type == "multi":
            loss = multi_task_loss(logits, annotations)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

    # report average loss across all batches
    # See https://stats.stackexchange.com/questions/201452/is-it-common-practice-to-minimize-the-mean-loss-over-the-batches-instead-of-the
    train_loss /= len(train_loader)
    return train_loss

def validate_one_epoch(model, val_loader, device, model_type, log_dict: dict) -> float:
    '''Validates the model on the validation set and returns the average validation loss over all batches. Will modify the given mutable log_dict to store the classification report and confusion matrix.'''
    model.eval()
    val_loss: float = 0.0
    with torch.no_grad():
        all_annotations = torch.tensor([]).to(device)
        all_predictions = torch.tensor([]).to(device)
        print("Starting batch loop")
        for input_ids, attention_mask, token_type_ids, annotations in tqdm(val_loader, desc="Validation batch"):
            inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "token_type_ids": token_type_ids.to(device)
            }
            annotations = annotations.to(device)

            logits = model(inputs)

            assert logits.shape[0] == len(annotations), f"Mismatch in batch size between logits and annotations. Got {logits.shape[0]} and {len(annotations)} respectively."

            if model_type == "base":
                raise NotImplementedError
                # majority_voted_annotations = majority_vote(annotations)
                # loss = nn.CrossEntropyLoss()(logits, majority_voted_annotations.squeeze())
            elif model_type == "multi":
                loss = multi_task_loss(logits, annotations)

            val_loss += loss.item()

            assert all_annotations.shape[0] == 0 or all_annotations.shape[1] == annotations.shape[1], f"Mismatch in number of annotators between all_annotations and annotations. Got {all_annotations.shape[1]} and {annotations.shape[1]} respectively."
            all_annotations = torch.cat((all_annotations, annotations), dim=0)
            all_predictions = torch.cat((all_predictions, logits_to_annotations(logits)), dim=0)

            assert all_annotations.shape == all_predictions.shape, f"Mismatch in shape between all_annotations and all_predictions. Got {all_annotations.shape} and {all_predictions.shape} respectively."
            assert all_annotations.ndim == 2, f"all_annotations should have 2 dimensions. Got {all_annotations.ndim} instead."
            assert all_predictions.ndim == 2, f"all_predictions should have 2 dimensions. Got {all_predictions.ndim} instead."
        print(f"Batch loop complete")

        # replace missing values with NaNs
        all_annotations, all_predictions = missing_to_nan(all_annotations, all_predictions)
        assert all_annotations.ndim == 2, f"all_annotations should have 2 dimensions. Got {all_annotations.ndim} instead."
        assert all_predictions.ndim == 2, f"all_predictions should have 2 dimensions. Got {all_predictions.ndim} instead."

        # get the aggregate values
        aggregate_annotations, aggregate_predictions = majority_vote(all_annotations.cpu()), majority_vote(all_predictions.cpu())
        assert aggregate_annotations.shape == aggregate_predictions.shape, f"Mismatch in shape between aggregate_annotations and aggregate_predictions. Got {aggregate_annotations.shape} and {aggregate_predictions.shape} respectively."
        assert aggregate_annotations.ndim == 1, f"aggregate_annotations should have 1 dimension. Got {aggregate_annotations.ndim} instead."
        assert aggregate_predictions.ndim == 1, f"aggregate_predictions should have 1 dimension. Got {aggregate_predictions.ndim} instead."

        # convert to list for classification_report and confusion_matrix
        aggregate_anotations_list, aggregate_predictions_list = aggregate_annotations.tolist(), aggregate_predictions.tolist()
        assert len(aggregate_anotations_list) == len(aggregate_predictions_list), f"Mismatch in length between aggregate_anotations_list and aggregate_predictions_list. Got {len(aggregate_anotations_list)} and {len(aggregate_predictions_list)} respectively."

        # get the aggregate classification report and confusion matrix
        log_dict[AGGREGATE_KEY] = {}
        log_dict[AGGREGATE_KEY][CLASSIFICATION_REPORT_KEY] = classification_report(aggregate_anotations_list, aggregate_predictions_list, output_dict=True, zero_division=0)
        log_dict[AGGREGATE_KEY][CONFUSION_MATRIX_KEY] = confusion_matrix(aggregate_anotations_list, aggregate_predictions_list)

        # flatten all the annotations and remove the nans for individual classification report and confusion matrix
        flattened_annotations_list, flattened_predictions_list = all_annotations.flatten().cpu().numpy(), all_predictions.flatten().cpu().numpy()
        flattened_annotations_list, flattened_predictions_list = flattened_annotations_list[~np.isnan(flattened_annotations_list)].tolist(), flattened_predictions_list[~np.isnan(flattened_predictions_list)].tolist()
        assert len(flattened_annotations_list) == len(flattened_predictions_list), f"Mismatch in length between flattened_annotations_list and flattened_predictions_list. Got {len(flattened_annotations_list)} and {len(flattened_predictions_list)} respectively."

        # get the individual classification report and confusion matrix
        log_dict[INDIVIDUAL_KEY] = {}
        log_dict[INDIVIDUAL_KEY][PREDICTIONS_KEY] = flattened_predictions_list
        log_dict[INDIVIDUAL_KEY][TRUE_LABELS_KEY] = flattened_annotations_list
        log_dict[INDIVIDUAL_KEY][CLASSIFICATION_REPORT_KEY] = classification_report(flattened_annotations_list, flattened_predictions_list, output_dict=True, zero_division=0)
        log_dict[INDIVIDUAL_KEY][CONFUSION_MATRIX_KEY] = confusion_matrix(flattened_annotations_list, flattened_predictions_list)

        # report average loss across all batches
        # See https://stats.stackexchange.com/questions/201452/is-it-common-practice-to-minimize-the-mean-loss-over-the-batches-instead-of-the
        val_loss /= len(val_loader)
    return val_loss

def train(model, train_loader, val_loader, device, model_type, num_epochs=10, logger: AdvancedLogger=None):
    '''Trains a model.'''
    if logger is None:
        logger = AdvancedLogger(AdvancedLoggerType.NONE)

    model, optimizer, scheduler = model_training_setup(model, train_loader, device, num_epochs)

    best_val_loss = float("inf")
    
    # check to see if val_loader.dataset.annotations is all -1s
    # if so, then we don't need to do validation
    do_validation = not (val_loader.dataset.annotations == -1).all()

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        log_dict = {}
        # use human readable epoch numbers
        log_dict["epoch"] = epoch + 1
        train_loss = train_one_epoch(model, train_loader, device, model_type, optimizer, scheduler)
        log_dict["train_loss"] = train_loss

        if do_validation:
            val_loss = validate_one_epoch(model, val_loader, device, model_type, log_dict)
            log_dict["val_loss"] = val_loss
        else:
            tqdm.write("\nNo validation data - skipping validation step")
        logger.log_dict(log_dict, epoch)

        if do_validation:
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Individual Weighted F1: {log_dict[INDIVIDUAL_KEY][CLASSIFICATION_REPORT_KEY]['weighted avg']['f1-score']}, Aggregate Weighted F1: {log_dict[AGGREGATE_KEY][CLASSIFICATION_REPORT_KEY]['weighted avg']['f1-score']}, Individual Accuracy: {log_dict[INDIVIDUAL_KEY][CLASSIFICATION_REPORT_KEY]['accuracy']}, Aggregate Accuracy: {log_dict[AGGREGATE_KEY][CLASSIFICATION_REPORT_KEY]['accuracy']}, Avg Batch Train Loss: {train_loss}, Avg Batch Val Loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # torch.save(model.state_dict(), "best_model.pt")
        else:
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Avg Batch Train Loss: {train_loss}")


if __name__ == "__main__":
    # test majority_vote
    annotations = torch.tensor([[1, 2, 3, 3, 5], [5, 5, 1, 1, 1]])
    print(majority_vote(annotations))