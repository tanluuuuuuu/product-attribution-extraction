from transformers import pipeline
from collections import defaultdict
import json
import argparse
from data_utils import preprocess_description
import pandas as pd

class NER():
    def __init__(
        self,
        model_checkpoint,
        mat_mapping = "data/mat_mapping.xlsx"
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = pipeline(
            "ner", model=model_checkpoint, aggregation_strategy="simple", device=0
        )
        self.mat_mapping = load_mat_mapping(mat_mapping)
        pass

    def predict(self):
        pass
    
    def load_mat_mapping(self, file_mapping):
        
        pass

