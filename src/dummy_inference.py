import json
from NER_class import NER

if __name__=='__main__':
    ner_model = NER(model_checkpoint="/home/tanluuuuuuu/Desktop/luunvt/direct_indirect/models/model_from_2023-12-18/roberta-base_09:01:27/best_f1", # Path to model checkpoint downloaded from Drive. Ex:  model/best_f1
                    mat_mapping="data/mat_mapping.xlsx") # Path to excel file contains material mapping dictionary. Ex: data/mat_mapping.xlsx
    
    description = '''
Solid Brazilian Cherry Wood legs with a black matte finish.
2 Full extension storage drawers with soft closing ball bearing steel runners.
A large open center compartment for center channel or sound bar speakers.
2 Concealed compartments with adjustable shelf on each side. The doors are soft closing.
Ships partially assembled
    '''
    predictions = ner_model.predict(description=description, # Bullet points or description as text for inference 
                                    text_preprocessed=False, # Set to False if text is not preprocessed
                                    map_level=3) # Mapping level from 1 to 3
    
    print("-"*100)            
    print(json.dumps(predictions, sort_keys=True, indent=4))                