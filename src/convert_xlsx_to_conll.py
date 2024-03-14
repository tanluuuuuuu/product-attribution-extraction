import os
import pandas as pd
import ast

def convert_locs(locs: str):
    locs = ast.literal_eval(locs)
    new_locs = []
    for loc in locs:
        new_locs.append( (int(loc[0]), int(loc[1]), loc[2]) )
    new_locs = sorted(new_locs)
    return new_locs

if __name__=="__main__":
    fixed_folder_data = "data/"
    modes = ['test', 'train']
    
    for mode in modes:
        file_xlsx = pd.read_excel(f"data/{mode}.xlsx")
        lines_txt = []
        for idx, row in file_xlsx.iterrows():
            sentence, locs, words = row
            locs = convert_locs(locs)
            words = ast.literal_eval(words)
            
            list_words = []
            list_labels = []
            bg = 0
            for loc in locs:
                start, end, label = loc
                span = sentence[bg:start].strip()
                
                if len(span) > 0:
                    list_word = span.split()
                    list_words.extend(list_word)
                    list_labels.extend(['O'] * len(list_word))
                
                list_word_label = sentence[start:end].strip().split()
                list_words.extend(list_word_label)
                list_labels.append(f"B-{label}")
                for i in range(1, len(list_word_label)):
                    list_labels.append(f"I-{label}")
                bg = end
                    
            if len(locs) == 0:
                list_word = sentence.strip().split()
                list_words.extend(list_word)
                list_labels.extend(['O'] * len(list_word))
            else:
                start, end, label = locs[-1]
                span = sentence[end:].strip()
                if len(span) > 0:
                    list_word = span.split()
                    list_words.extend(list_word)
                    list_labels.extend(['O'] * len(list_word))
                        
            for word, label in zip(list_words, list_labels):
                lines_txt.append(f"{word} {label}\n")
            lines_txt.append("\n")
            
        new_file_name = f"{mode}.txt"
        f_out = open(os.path.join(fixed_folder_data, new_file_name), "w")
        for line in lines_txt:
            f_out.write(line)
            