from transformers import pipeline
from collections import defaultdict
import json
import argparse
from data_utils import preprocess_description
from NER_class import NER

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", help="Link model", type=str, required=True)
    parser.add_argument('-t', "--text", help="Text to extract material and color", type=str, required=True)
    args = parser.parse_args()
    
    ner_model = NER(model_checkpoint="Set model path here", # Path to model checkpoint downloaded from Drive. Ex:  model/best_f1
                    mat_mapping="../data/mat_mapping.xlsx") # Path to excel file contains material mapping dictionary
    
    predictions = ner_model.predict(description='Some text here', # Bullet points or description as text for inference 
                                    text_preprocessed=False, # Set to False if text is not preprocessed
                                    map_level=3) # Mapping level from 1 to 3
    
    print("-"*100)            
    print(json.dumps(predictions, sort_keys=True, indent=4))                