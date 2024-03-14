from HGN import HGNER, HGNER_wrapper
from transformers import AutoTokenizer, pipeline
import torch
import argparse
from pipeline import postprocess
import pandas as pd
from tqdm import tqdm

def do_anything_I_want(tokenizer, model_best, id2label):
    df = pd.read_csv("/home/luunvt/WORK_DIR/luunvt/direct_indirect/data/asin_info_202401151145.csv")
    df = df.dropna()
    list_materials = list(set(df['material'].to_list()))

    list_confirmed = []
    list_not_confirmed = []
    for material in tqdm(list_materials):
        new_material = [f"Material: {material}"]
        res = postprocess(new_material, tokenizer, model_best, id2label)
        if len(res) > 0:
            list_confirmed.append(material)
        else:
            list_not_confirmed.append(material)
    list_confirmed = list(set(list_confirmed))
    list_not_confirmed = list(set(list_not_confirmed))

    mat_txt = open("/home/luunvt/WORK_DIR/luunvt/direct_indirect/src/HGN/output/list_mat.txt", 'w')
    not_mat_txt = open("/home/luunvt/WORK_DIR/luunvt/direct_indirect/src/HGN/output/list_not_mat.txt", 'w')
    
    for mat in list_confirmed:
        mat_txt.write(f"{mat}\n")

    for mat in list_not_confirmed:
        not_mat_txt.write(f"{mat}\n")


if __name__ == '__main__':
    weight_path = "/home/luunvt/WORK_DIR/luunvt/direct_indirect/src/HGN/output/roberta-large_dot_multi_window/epoch_15.pth"

    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model",
                        default='roberta-base',
                        type=str)
    parser.add_argument("--use_bilstm",
                        action='store_false')
    parser.add_argument("--use_multiple_window",
                        action='store_false')
    parser.add_argument("--connect_type",
                        default='dot-att',
                        type=str)
    parser.add_argument("--windows_list",
                        default='1qq3qq5qq7',
                        type=str)
    parser.add_argument("--use_crf",
                    action='store_true',
                    help="Whether use crf")
    args = parser.parse_args()

    id2label = {
        1: "O",
        2: "B-MAT",
        3: "B-COLOR",
        4: "I-MAT",
        5: "I-COLOR",
    }

    model_best = HGNER(
        args,
        hidden_dropout_prob=0.1,
        num_labels=6,
        windows_list = [1, 3, 5, 7]
    )
    model_best.load_state_dict(torch.load(weight_path))
    model_best = model_best.to('cuda').eval()

    tokenizer = AutoTokenizer.from_pretrained('xlnet-large-cased')

    text = "Material: Neoprene."
    res = postprocess(text, tokenizer, model_best, id2label)
    print(res)
    # do_anything_I_want(tokenizer, model_best, id2label)