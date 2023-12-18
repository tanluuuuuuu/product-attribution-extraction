from transformers import pipeline
from collections import defaultdict
import json
import argparse
from data_utils import preprocess_description
from NER_class import NER
import pandas as pd
from tqdm import tqdm
import ast
import os
import pickle
import shutil

MODEL_CHECKPOINT = "Model checkpoint path here" # Path to model checkpoint downloaded from Drive. Ex:  model/best_f1
MAT_MAPPING = "Material mapping path here" # Path to excel file contains material mapping dictionary. Ex: data/mat_mapping.xlsx

MAP_LEVEL = 3
if __name__ == '__main__':
    ner_model = NER(model_checkpoint=MODEL_CHECKPOINT,
                    mat_mapping=MAT_MAPPING)
    
    list_direct, list_indirect = ner_model.get_direct_indirect(
        focus_descriptions=dict,
        candidate_descriptions=dict,
        map_level=MAP_LEVEL
    )
    
    
    