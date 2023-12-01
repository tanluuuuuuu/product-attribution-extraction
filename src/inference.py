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
    
    ner_model = NER(args.model)
    predictions = ner_model.predict(args.text)
    
    print("-"*100)            
    print(json.dumps(predictions, sort_keys=True, indent=4))                