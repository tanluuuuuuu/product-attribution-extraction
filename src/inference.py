import json
from NER_class import NER

if __name__=='__main__':
    ner_model = NER(model_checkpoint="Set model path here", # Path to model checkpoint downloaded from Drive. Ex:  model/best_f1
                    mat_mapping="Set mat mapping file here") # Path to excel file contains material mapping dictionary. Ex: data/mat_mapping.xlsx
    
    predictions = ner_model.predict(description='Some text here', # Bullet points or description as text for inference 
                                    text_preprocessed=False, # Set to False if text is not preprocessed
                                    map_level=3) # Mapping level from 1 to 3
    
    print("-"*100)            
    print(json.dumps(predictions, sort_keys=True, indent=4))                