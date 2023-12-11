# make sure your downloaded the english model with "python -m spacy download en"

import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(u"gas")

for token in doc:
    print(token, token.lemma, token.lemma_)