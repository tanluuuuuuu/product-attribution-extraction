from transformers import pipeline
from collections import defaultdict
import json
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", help="Link model", type=str, required=True)
    parser.add_argument('-t', "--text", help="Text to extract material and color", type=str, required=True)
    args = parser.parse_args()
    
    model_checkpoint = args.model
    token_classifier = pipeline(
        "ner", model=model_checkpoint, aggregation_strategy="simple", device=0
    )

    # Replace this description
    description = args.text

    def preprocess_description(description, words_need_removed = []):
        # add space to string
        single_description = description.strip()
        new_description = []
        last_special = -1
        for idx, letter in enumerate(single_description):
            if not (('a' <= letter and letter <= 'z') or ('A' <= letter and letter <= 'Z') or ('0' <= letter and letter <= '9') or letter == ' '):
                pretext = single_description[last_special + 1:idx].strip()
                if pretext != '' and pretext != ' ':
                    new_description.append(pretext)
                new_description.append(letter.strip())
                last_special = idx
            if idx == len(single_description) - 1:
                new_description.append(
                    single_description[last_special + 1:idx + 1].strip())
        new_description = " ".join(new_description)
        
        # Remove words from string: brand name,...
        querywords = new_description.split(" ")
        words_need_removed = [x.lower() for x in words_need_removed]
        resultwords  = [word for word in querywords if word.lower() not in words_need_removed]
        result = ' '.join(resultwords)
        
        return result


    high_score_ans = defaultdict(set)
    bullet_points = description.split("\n")
    for bullet_point in bullet_points:
        bullet_point = preprocess_description(bullet_point)

        if bullet_point != "":
            print(bullet_point)

            results = token_classifier(bullet_point)
            for res in results:
                if res['word'].lower().strip() in ['durable', 'strong', 'heavy-duty', 'heavy duty', 'stability', 'versatile', 'comfortable']:
                    continue
                group = res['entity_group']
                if res['score'] >= 0.9:
                    high_score_ans[group].add(res['word'].strip().lower())
                    
    new_high_score_ans = defaultdict(list)
    for key_dict in high_score_ans.keys():
        new_high_score_ans[key_dict] = list(high_score_ans[key_dict])

    print("-"*100)            
    print(json.dumps(new_high_score_ans, sort_keys=True, indent=4))                