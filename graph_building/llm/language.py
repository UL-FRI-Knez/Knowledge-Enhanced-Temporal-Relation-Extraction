from __future__ import unicode_literals
import spacy,en_core_web_sm
import textacy
nlp = en_core_web_sm.load()
sentence = 'The cat sat on the mat. He dog jumped into the water. The author is writing a book.'
pattern = [{'POS': 'VERB', 'OP': '?'},
           {'POS': 'ADV', 'OP': '*'},
           {'POS': 'VERB', 'OP': '+'}]
doc = textacy.make_spacy_doc(sentence, lang='en_core_web_sm')
lists = textacy.extract.matches(doc, pattern)
for list in lists:
    print(list.text)