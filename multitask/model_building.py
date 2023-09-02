import torch
import torch.nn as nn
from transformers import AutoModel
import numpy as np

class RatingModel(nn.Module):
    BASE_MODEL = "base"
    MULTI_MODEL = "multi"

    def __init__(self, encoder_model, model_type: str, num_annotators, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_model, output_hidden_states=True)
        self.model_type: str = model_type

        assert model_type in (self.BASE_MODEL, self.MULTI_MODEL), f"Invalid model_type. Must be one of {self.BASE_MODEL} or {self.MULTI_MODEL}. Got {model_type} instead."
        if model_type == self.BASE_MODEL:
            self.predictor = nn.Linear(self.encoder.config.hidden_size, num_classes)
        elif model_type == self.MULTI_MODEL:
            self.predictors = nn.ModuleList([nn.Linear(self.encoder.config.hidden_size, num_classes) for _ in range(num_annotators)])

    def forward(self, inputs):
        assert inputs["input_ids"].ndim == 2, "Input must have 2 dimensions (batch_size, sequence_length). Got {inputs['input_ids'].ndim} instead."
        # outputs = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
        outputs = self.encoder(**inputs)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        if self.model_type == self.BASE_MODEL:
            return self.predictor(cls_output)
        elif self.model_type == self.MULTI_MODEL:
            # return np.stack([predictor(cls_output) for predictor in self.predictors], axis=1)
            # can't use numpy so use torch instead
            return torch.stack([predictor(cls_output) for predictor in self.predictors], dim=1)