from transformers import pipeline
from collections import defaultdict, Counter
import json
import argparse
from data_utils import preprocess_description
import pandas as pd
import torch
import re
from nltk.stem import PorterStemmer
from collections import defaultdict
import spacy
from thefuzz import fuzz
from thefuzz import process


class NER():
    def __init__(
        self,
        model_checkpoint,
        mat_mapping="data/mat_mapping.xlsx",
    ):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = pipeline(
            "ner", model=model_checkpoint, aggregation_strategy="simple", device=0
        )
        self.mat_mapping = self.load_mat_mapping(mat_mapping)
        self.mat_mapping_lv1 = self.load_mat_lv1(self.mat_mapping)
        self.ps = PorterStemmer()
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, description, text_preprocessed=False, map_level=3):
        high_score_ans = defaultdict(set)
        bullet_points = description.split("\n")
        for bullet_point in bullet_points:
            if not text_preprocessed:
                bullet_point = self.preprocess_description(bullet_point)

            if bullet_point != "":
                results = self.model(bullet_point)
                for res in results:
                    group = res['entity_group']
                    if res['score'] >= 0.8:
                        word = [res['word'].strip().lower()]
                        if group == 'MAT':
                            word = self.find_nearest_mat(word)
                            mapped_word = self.map_word(word, map_level)
                            word = self.lemmatize_words(mapped_word)
                        high_score_ans[group].update(word)

        new_high_score_ans = defaultdict(list)
        for key_dict in high_score_ans.keys():
            new_high_score_ans[key_dict] = sorted(
                list(set(high_score_ans[key_dict])))
        return new_high_score_ans

    def find_nearest_mat(self, word):
        sgl_word = word[0]
        nearest_lv1 = process.extractOne(sgl_word, self.mat_mapping.keys())
        if nearest_lv1[1] >= 85:
            return nearest_lv1[0]
        else:
            nearest_lv2 = process.extractOne(sgl_word, self.mat_mapping_lv1.keys())
            if nearest_lv2[1] >= 85:
                return nearest_lv2[0]
            else:
                return sgl_word

    def load_mat_mapping(self, file_mapping):
        file_excel = pd.read_excel(file_mapping)
        mapping = dict()
        for idx, row in file_excel.iterrows():
            mapping[row['input']] = [row['output_1'],
                                     row['output_2'], row['output_3']]
        return mapping

    def load_mat_lv1(self, mat_mapping):
        mat_lv1 = dict()
        for key in mat_mapping:
            mats = mat_mapping[key]
            mat_lv1[mats[0]] = mats[1:]
        return mat_lv1

    def preprocess_description(self, description, words_need_removed=[]):
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
        resultwords = [word for word in querywords if word.lower()
                       not in words_need_removed]
        result = ' '.join(resultwords)
        return result

    def map_word(self, word, map_level):
        map_level -= 1
        if map_level >= 1:
            return self._map_word_general(word, map_level)
        else:
            return self._map_word_specific(word, map_level)

    def _map_word_general(self, word, map_level):
        if word in self.mat_mapping:
            return [self.mat_mapping[word][map_level]]
        if word in self.mat_mapping_lv1:
            return [self.mat_mapping_lv1[word][map_level - 1]]
        small_words = word.split(" ")
        res = []
        for small_word in small_words:
            if small_word in self.mat_mapping:
                mapped = self.mat_mapping[small_word][map_level]
            elif small_word in self.mat_mapping_lv1:
                mapped = self.mat_mapping_lv1[small_word][map_level - 1]
            else:
                continue
            res.append(mapped)
        res = list(set(res))
        return res

    def _map_word_specific(self, word, map_level):
        if word in self.mat_mapping:
            mapped = self.mat_mapping[word][map_level]
            return [mapped]
        else:
            return [word]

    def stemming_word(self, word):
        new_words = [self.ps.stem(w) for w in word.split(" ")]
        return " ".join(new_words)

    def lemmatize_words(self, words):
        list_new_words = []
        for word in words:
            doc = self.nlp(word)
            new_word = []
            for token in doc:
                new_word.append(token.lemma_)
            list_new_words.append(" ".join(new_word))
        return list_new_words

    def split_by_materials(
        self,
        product_descriptions: dict,
        output_dir: str,
        image_folder_path: str
    ) -> None:
        """
        Split products into difference materials

        Parameters
        ----------
        product_descriptions: dictionary
            Dictionary of product descriptions. 
            Example format: 
            {
                'B08SQ66QRL': "Material: iron"    
            }
        output_dir: string
            Path of folder output to save split results.
        image_folder_path: string
            Path of image folder to load images to save results.

        Returns
        -------
        None 
        """
        for asin in tqdm(product_descriptions):
            descriptions = product_descriptions[asin]

            if len(descriptions) <= 1:
                print(f"NULL description at asin {asin}")
                continue

            predictions = self.model.predict(descriptions,
                                             text_preprocessed=False,
                                             map_level=3)

            try:
                folder_name = "_".join(predictions['MAT'])
                if len(folder_name) == 0:
                    folder_name = "no_mat"
                if not os.path.exists(os.path.join(output_dir, folder_name)):
                    os.makedirs(os.path.join(output_dir, folder_name))
                folder_output_path = os.path.join(output_dir, folder_name)
                image_src = os.path.join(image_folder_path, f"{asin}.jpg")
                image_des = os.path.join(
                    output_dir, folder_name, f"{asin}.jpg")
                shutil.copy(image_src, image_des)
            except:
                print(f"Fail to save image at asin {asin}")
                continue

        print("***** SPLIT BY MATERIAL DONE!!! *****")

    def _get_direct_indirect_by_materials(
        self,
        list_material: list,
        candidate_descriptions: dict,
        output_dir: str,
        map_level: int,
    ) -> None:
        list_direct_asin = []
        list_indirect_asin = []
        for asin in tqdm(candidate_descriptions):
            descriptions = candidate_descriptions[asin]

            if len(descriptions) <= 1:
                print(f"NULL description at asin {asin}")
                continue

            predictions = self.model.predict(descriptions,
                                             text_preprocessed=False,
                                             map_level=map_level)

            mat_prediction = predictions['MAT']
            if len(set(list_material).intersection(set(mat_prediction))) >= 1:
                list_direct_asin.append(asin)
            else:
                list_indirect_asin.append(asin)
        return list_direct_asin, list_indirect_asin

    def get_direct_indirect(
        self,
        focus_descriptions: dict,
        candidate_descriptions: dict,
        map_level: int,
    ) -> tuple:
        """
        Split products into difference materials

        Parameters
        ----------
        focus_descriptions: dictionary
            Dictionary of focus descriptions. 
            Example format: 
            {
                'B08SQ66QRL': "Material: iron"    
            }
        candidate_descriptions: dictionary
            Dictionary of candidate descriptions, same format as focus_descriptions
        map_level: int,
            Map level, range from 1 to 3

        Returns
        -------
        Tuple:
            - List of direct asins
            - List of indirect asins 
        """

        print("Get focus materials")
        list_focus_materials = []
        for asin in tqdm(focus_descriptions):
            descriptions = focus_descriptions[asin]

            if not isinstance(descriptions, str):
                print(f"NULL description at focus asin {asin}")
                continue

            predictions = self.predict(descriptions,
                                       text_preprocessed=False,
                                       map_level=map_level)
            list_focus_materials.extend(predictions['MAT'])
        list_focus_materials = sorted(list(set(list_focus_materials)))

        # Get list direct indirect
        print("Get list direct indirect")
        list_direct, list_indirect = self._get_direct_indirect_by_materials(
            list_material=list_focus_materials,
            candidate_descriptions=candidate_descriptions,
            output_dir=output_dir,
            map_level=map_level
        )
        return list_direct, list_indirect
