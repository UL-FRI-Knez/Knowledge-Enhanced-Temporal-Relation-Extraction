import torch
from transformers import BertTokenizer,BertModel
from transformers import Data2VecTextConfig, Data2VecTextModel

from custom_datasets.common import lock
from date2vec.date_embedding import compute_date_embedding

tokenizer = None
model = None
def sentence_embedding(text, type='bert'):
    global tokenizer, model
    if text is None or len(text.strip()) == 0:
        text = "empty"
    if type == 'bert':
        with lock:
            if tokenizer is None:
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            if model is None:
                model = BertModel.from_pretrained("bert-base-uncased")
        tokens = tokenizer(text, return_tensors='pt', max_length=512)
        output = model(**tokens)
        last_hidden_state, pooler_output = output[0], output[1]
        return pooler_output.detach()

def date_embedding(date):
    # Initializing a Data2VecText facebook/data2vec-text-base style configuration
    return compute_date_embedding(year=date[0], month=date[1], day=date[2]).detach()

if __name__ == '__main__':
    text = "This is a sample sentence."
    date_embedding(text)
    print(sentence_embedding(text))