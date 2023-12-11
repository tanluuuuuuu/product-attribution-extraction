from transformers import pipeline
from collections import defaultdict
import json
import argparse
from data_utils import preprocess_description
import pandas as pd
import torch
import re
from nltk.stem import PorterStemmer
from collections import defaultdict

class NER():
    def __init__(
        self,
        model_checkpoint,
        mat_mapping = "data/mat_mapping.xlsx"
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = pipeline(
            "ner", model=model_checkpoint, aggregation_strategy="simple", device=0
        )
        self.mat_mapping = self.load_mat_mapping(mat_mapping)
        self.ps = PorterStemmer()

    def predict(self, description):
        high_score_ans = defaultdict(set)
        bullet_points = description.split("\n")
        for bullet_point in bullet_points:
            bullet_point = self.preprocess_description(bullet_point)

            if bullet_point != "":
                results = self.model(bullet_point)
                for res in results:
                    group = res['entity_group']
                    if res['score'] >= 0.9:
                        word = res['word'].strip().lower()
                        mapped_word = self.map_word(word)
                        word_stemmed = self.stemming_word(mapped_word)
                        high_score_ans[group].add(word_stemmed)
                        
        new_high_score_ans = defaultdict(list)
        for key_dict in high_score_ans.keys():
            new_high_score_ans[key_dict] = list(high_score_ans[key_dict])
        return new_high_score_ans
    
    def load_mat_mapping(self, file_mapping):
        file_excel = pd.read_excel(file_mapping)
        mapping = dict()
        for idx, row in file_excel.iterrows():
            mapping[row['input']] = row['output']
        return mapping
    
    def preprocess_description(self, description, words_need_removed = []):
        # add space to string
        single_description = description.strip()
        pattern = "[^a-zA-Z0-9\s]"
        matches = re.finditer(pattern, single_description)
        
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
    
    def map_word(self, word):
        small_words = word.split(" ")
        res = []
        for small_word in small_words:
            mapped = self.mat_mapping[small_word] if small_word in self.mat_mapping else small_word
            res.append(mapped)
        res = " ".join(res)
        return res
    
    def stemming_word(self, word):
        new_words = [self.ps.stem(w) for w in word.split(" ")]
        return " ".join(new_words)