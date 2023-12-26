from transformers import RobertaForTokenClassification
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
import os
from datetime import datetime

def setup_model(ner_tags_2_number, number_2_ner_tags):
    model = RobertaForTokenClassification.from_pretrained(
        "roberta-base",
        id2label=number_2_ner_tags,
        label2id=ner_tags_2_number,
        ignore_mismatched_sizes=True
    )
    return model
    
def save_model(
    output_dir, 
    folder_name, 
    tokenizer, 
    unwrapped_model, 
    save_func
):
    """
    """
    output_ckpt = os.path.join(output_dir, folder_name)
    tokenizer.save_pretrained(output_ckpt)
    unwrapped_model.save_pretrained(
        output_ckpt, save_function=save_func)
    pass
    
def get_output_dir():
    date_time = datetime.now()
    format_date = date_time.strftime('%Y-%m-%d')
    format_time = date_time.strftime('%H:%M:%S')
    print(f"Date: {format_date}")
    print(f"Time: {format_time}")

    output_dir = f"models/model_from_{format_date}/roberta-base_{format_time}"
    print("Output dir: ", output_dir)
    return output_dir