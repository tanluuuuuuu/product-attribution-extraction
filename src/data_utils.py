from transformers import RobertaTokenizerFast
import pandas as pd
from datasets import Dataset
from typing import Dict, List
import ast
from datasets import Dataset, DatasetDict, ClassLabel, Features, Value, Sequence
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast
import re

def load_data(
    excel_path: str,
    tokenizer: RobertaTokenizerFast,
    ner_tags_2_number: Dict
) -> Dataset:
    """
    Load data from excel file for training and/or testing, contains 'sentence', 'locs', 'words'
    Parameters
    ----------
    excel_path: string
        An input of path to excel file
    tokenizer: RobertaTokenizerFast
        An Roberta tokenizer loaded from hugginface hub
    ner_tags_2_number: Dict
        An dictionary mapping ner tags B- I- to corresponding id

    Returns
    -------
    Dataset
        An Huggingface dataset produced from excel file.
    """

    raw_dataset = pd.read_excel(excel_path)

    list_input_ids = []
    list_attention_mask = []
    list_labels = []

    for index, row in raw_dataset.iterrows():
        label = assign_ner_tags_roberta(row, tokenizer, ner_tags_2_number)
        token_input = tokenizer(row['sentence'])
        list_input_ids.append(token_input['input_ids'])
        list_attention_mask.append(token_input['attention_mask'])
        list_labels.append(label)

    tokenized_datasets = pd.DataFrame()
    tokenized_datasets['input_ids'] = pd.Series(list_input_ids)
    tokenized_datasets['attention_mask'] = pd.Series(list_attention_mask)
    tokenized_datasets['labels'] = pd.Series(list_labels)

    dataset = Dataset.from_pandas(tokenized_datasets)
    return dataset

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[processed_ner_tags[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [processed_ner_tags[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

def assign_ner_tags_roberta(
    example: pd.Series,
    tokenizer: RobertaTokenizerFast,
    ner_tags_2_number: Dict
) -> List[int]:
    """
    Generate label w.r.t input ids of an sample sentence.
    Parameters
    ----------
    example: pandas Series, shape = (*)
        An input of row in excel file
    tokenizer: RobertaTokenizerFast
        An Roberta tokenizer loaded from hugginface hub
    ner_tags_2_number: Dict
        An dictionary mapping ner tags B- I- to corresponding id

    Returns
    -------
    List, shape = (*)
        An array of number represent label w.r.t input ids.
    """
    token_input = tokenizer(example['sentence'])
    example['tokens'] = tokenizer.convert_ids_to_tokens(
        token_input['input_ids'])

    ner_tags = [0 for token in example['tokens']]
    locs = ast.literal_eval(example['locs'])

    locs = [(int(loc[0]), int(loc[1]), loc[2]) for loc in locs]
    locs = sorted(locs)
    bg_id = 1
    pre_loc = 0
    text = example['sentence']
    for loc in locs:
        loc0 = int(loc[0])
        loc1 = int(loc[1])

        pre_text = text[pre_loc:loc0]
        if 0 < loc0 and len(pre_text) > 0 and pre_text[-1] == ' ':
            pre_text = text[pre_loc:loc0 - 1]
        token_input = tokenizer(pre_text)
        pre_token = tokenizer.convert_ids_to_tokens(
            token_input['input_ids'])
        bg_id = bg_id + len(pre_token) - 2
        pre_loc = loc1

        word = example['sentence'][loc0: loc1].strip()
        if loc0 > 0 and example['sentence'][loc0 - 1] == ' ':
            word = " " + word
        token_input = tokenizer(word)
        word_token = tokenizer.convert_ids_to_tokens(token_input['input_ids'])

        label_number = ner_tags_2_number[f"B-{loc[2]}"]
        ner_tags[bg_id] = label_number
        bg_id += 1
        for idx in range(bg_id, bg_id + len(word_token) - 3):
            ner_tags[idx] = label_number + 1
        bg_id = bg_id + len(word_token) - 3

    ner_tags[0] = -100
    ner_tags[-1] = -100
    return ner_tags

def postprocess(predictions, labels, processed_ner_tags):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[processed_ner_tags[l]
                    for l in label if l != -100] for label in labels]
    true_predictions = [
        [processed_ner_tags[p]
            for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions

def preprocess_description(description, words_need_removed = []):
    # add space to string
    single_description = description.strip()
    pattern = "[^a-zA-Z0-9\s]"
    matches = re.finditer(pattern, text)
    
    new_description = []
    before = 0
    for match in matches:
        new_description.append(description[before:match.start()].strip())
        new_description.append(description[match.start():match.end()])
        before = match.end()
    new_description.append(description[before:].strip())
    new_description = " ".join(new_description).strip()
    
    # TODO: Remove words from string: brand name,...
    querywords = new_description.split(" ")
    words_need_removed = [x.lower() for x in words_need_removed]
    resultwords  = [word for word in querywords if word.lower() not in words_need_removed]
    result = ' '.join(resultwords)
    return result
