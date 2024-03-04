import json
from NER_class import NER
import argparse

def argParse():
    parser = argparse.ArgumentParser(description='Attribution extraction')
    parser.add_argument('-m', '--model', type=str, help='Model path')
    parser.add_argument('-f', '--file_mapping', type=str, help='Excel path to map materials')
    parser.add_argument('-t', '--text', default='This product made from iron',
                        help='Text to inference')

    args = parser.parse_args()
    return args

if __name__=='__main__':
    
    args = argParse()
    ner_model = NER(model_checkpoint=args.model, # Path to model checkpoint downloaded from Drive. Ex:  model/best_f1
                    mat_mapping=args.file_mapping) # Path to excel file contains material mapping dictionary. Ex: data/mat_mapping.xlsx
    
    predictions = ner_model.predict(description=args.text, # Bullet points or description as text for inference 
                                    text_preprocessed=False, # Set to False if text is not preprocessed
                                    map_level=3) # Mapping level from 1 to 3
    
    print("-"*100)            
    print(json.dumps(predictions, sort_keys=True, indent=4))                