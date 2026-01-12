import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class RelationDetector(nn.Module):
    def __init__(self, number_of_relations=2, dropout=0.2, deeper_network=False, pooling_strategy='cls'):
        super(RelationDetector, self).__init__()
        self.EntityBert = AutoModel.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT")
        self.tokenizer = AutoTokenizer.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT")
        self.pooling_strategy = pooling_strategy
        self.criterion = nn.CrossEntropyLoss()
        for param in self.EntityBert.parameters():
            param.requires_grad = False

        if pooling_strategy == 'both_events':
            input_size = 768 * 2
        else:
            input_size = 768

        self.dimension_reduction = nn.Linear(input_size, 64)
        self.post_layers = nn.Linear(64, number_of_relations)
        self.softmax = nn.Softmax(dim=1)

    # def forward(self, text, event1_start, event1_end, event2_start, event2_end):
    def forward(self, data, labels=None, return_embedding=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data.to(device)
        text, event1_start, event1_end, event2_start, event2_end = data.text, data.event1_start, data.event1_end, data.event2_start, data.event2_end
        text = list(text)
        # for i in range(len(text)):
        #     text[i], event1_start[i], event1_end[i], event2_start[i], event2_end[i] = self.add_event_tokens(text[i], int(event1_start[i]), int(event1_end[i]), int(event2_start[i]), int(event2_end[i]))

        tokens = self.tokenizer(text, return_tensors="pt", max_length=100, padding='max_length', truncation=True)
        tokens.to(self.EntityBert.device)
        x = self.EntityBert(**tokens)
        if self.pooling_strategy == 'cls':
            bert_output = x['last_hidden_state'][:, 0, :]
        elif self.pooling_strategy == 'pool':
            bert_output = x['pooler_output']
        elif self.pooling_strategy == 'both_events':
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



        x = self.dimension_reduction(bert_output)
        if return_embedding:
            return x
        x = self.post_layers(x)
        x = self.softmax(x)
        loss = self.criterion(x, labels)
        return {"loss": loss, "logits": x}

    def add_event_tokens(self, text, event1_start, event1_end, event2_start, event2_end):
        tag_start1, tag_start2, tag_end1, tag_end2 = "<e1>", "<e2>", "</e1>", "</e2>"
        # tag_start1, tag_start2, tag_end1, tag_end2 = "<e>", "<e>", "</e>", "</e>"
        text = text[:event1_start] + tag_start1 + text[event1_start:]
        if event1_end >= event1_start:
            event1_end += len(tag_start1)
        if event2_start >= event1_start:
            event2_start += len(tag_start1)
        if event2_end >= event1_start:
            event2_end += len(tag_start1)
        if event1_start >= event1_start:
            event1_start += len(tag_start1)

        text = text[:event1_end] + tag_end1 + text[event1_end:]
        if event1_start > event1_end:
            event1_start += len(tag_end1)
        if event2_start > event1_end:
            event2_start += len(tag_end1)
        if event2_end > event1_end:
            event2_end += len(tag_end1)

        if max(event1_start, event2_start) < min(event1_end, event2_end):
            return text, event1_start, event1_end, event1_start, event1_end

        text = text[:event2_start] + tag_start2 + text[event2_start:]
        if event1_start >= event2_start:
            event1_start += len(tag_start2)
        if event1_end >= event2_start:
            event1_end += len(tag_start2)
        if event2_end >= event2_start:
            event2_end += len(tag_start2)
        if event2_start >= event2_start:
            event2_start += len(tag_start2)

        text = text[:event2_end] + tag_end2 + text[event2_end:]
        if event1_start >= event2_end:
            event1_start += len(tag_end2)
        if event1_end >= event2_end:
            event1_end += len(tag_end2)
        if event2_start >= event2_end:
            event2_start += len(tag_end2)
        return text, event1_start, event1_end, event2_start, event2_end