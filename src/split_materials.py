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

PRODUCT_INFO = "Excel path of Product information here" # Path to excel file contains product bulletpoints or description. Ex: data/kettlebell.xlsx
IMAGE_FOLDER = "Folder path contains images here" # Path to folder contains product images.
OUTPUT_PATH = "Folder path here"

if __name__ == '__main__':
    assert len(os.listdir(IMAGE_FOLDER)) > 0, "Image folder should not be empty"
    
    ner_model = NER(model_checkpoint=MODEL_CHECKPOINT,
                    mat_mapping=MAT_MAPPING)
    
    ner_model.split_by_materials(
        preprocess_description=dict,
        output_dir=OUTPUT_PATH,
        image_folder_path=IMAGE_FOLDER
    )