import json
import argparse
from transformers import pipeline

def argParse():
    parser = argparse.ArgumentParser(description='Attribution extraction')
    parser.add_argument('-m', '--model_path', type=str, help='Model path')
    parser.add_argument('-t', '--text', default='This product made from iron',
                        help='Text to inference')

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = argParse()
    token_classifier = pipeline(
    "token-classification", model=args.model_path, aggregation_strategy="simple"
    )
    
    predictions = token_classifier(args.text)
    
    print("-"*100)            
    print(json.dumps(predictions, sort_keys=True, indent=4))                