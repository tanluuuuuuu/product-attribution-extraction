from transformers import pipeline
from collections import defaultdict
import json
import argparse
from data_utils import preprocess_description
from NER_class import NER

if __name__ == '__main__':
    ner_model = NER("/home/tanluuuuuuu/Desktop/luunvt/direct_indirect/models/model_from_2023-12-01/roberta-base_14:11:39/best_f1")

    text = '''
GAS GRILL: Powered by liquid propane gas, this grill runs on a 20-pound tank, not included. It reaches cooking temperatures quickly, delivers high temps and maintains even heat.
COOKING AREA: Grill up to 8 burgers, 4 chicken breasts or 2 steaks on the 280-square-inch grate.
BTU RATING: This grill has a 20,000 BTU rating with two 10,000 BTU burners. *BTU is a measurement of energy used by your grill and is not related to how hot it will get. Grills with lower BTUs burn less fuel to cook your food.
GRATES: These porcelain-coated wire grates offer fast warm-up and heat recovery time. They’re lightweight, making them easy to lift when removing to clean and economical to replace when it’s time.
SIDE SHELVES: Two large side shelves provide ample space to set plates, tools, sauces and rubs while you grill.
IGNITION: Push-button ignition ensures reliable starts every time. Simply turn the control knob, press the ignition button and the burners are lit.
CONTROL KNOBS: Control knobs raise and lower the flame for each burner. Turn clockwise to increase the heat and counterclockwise to decrease.
    '''
    predictions = ner_model.predict(text)

    print("-"*100)
    print(json.dumps(predictions, sort_keys=True, indent=4))
