import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class BaselineBERT(nn.Module):
    def __init__(self, number_of_relations=3, dropout=0.2, deeper_network=False, pooling_strategy='cls'):
        super(BaselineBERT, self).__init__()
        self.EntityBert = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.pooling_strategy = pooling_strategy
        for param in self.EntityBert.parameters():
            param.requires_grad = False

        if pooling_strategy == 'both_events':
            input_size = 768 * 2
        else:
            input_size = 768

        if deeper_network:
            self.post_layers = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_size, 256),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, number_of_relations))
        else:
            self.post_layers = nn.Linear(input_size, number_of_relations)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()

    # def forward(self, text, event1_start, event1_end, event2_start, event2_end):
    def forward(self, data, labels, return_embedding=False):
        text, event1_start, event1_end, event2_start, event2_end = data.text, data.event1_start, data.event1_end, data.event2_start, data.event2_end
        text = list(text)
        tokens = self.tokenizer(text, return_tensors="pt", max_length=100, padding='max_length', truncation=True)
        tokens.to(self.EntityBert.device)
        x = self.EntityBert(**tokens)
        if self.pooling_strategy == 'cls':
            bert_output = x['last_hidden_state'][:, 0, :]
        elif self.pooling_strategy == 'pool':
            bert_output = x['pooler_output']
        else:
            bert_output = []
            for i, layer in enumerate(x['last_hidden_state']):
                event1_emb = x['last_hidden_state'][i][
                             tokens[i].char_to_token(event1_start[i]):tokens[i].char_to_token(event1_end[i])]
                event1_emb = torch.mean(event1_emb, 0)
                event2_emb = x['last_hidden_state'][i][
                             tokens[i].char_to_token(event2_start[i]):tokens[i].char_to_token(event2_end[i])]
                event2_emb = torch.mean(event2_emb, 0)
                output = torch.cat((event1_emb, event2_emb))
                bert_output.append(output)
            bert_output = torch.stack(bert_output)

        if return_embedding:
            return bert_output

        x = self.post_layers(bert_output)
        x = self.softmax(x)
        loss = self.criterion(x, labels)
        return {"loss": loss, "logits": x}