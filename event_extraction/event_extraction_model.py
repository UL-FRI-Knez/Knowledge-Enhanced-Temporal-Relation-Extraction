import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

class EventExtraction(nn.Module):
    def __init__(self, tokenizer):
        super(EventExtraction, self).__init__()
        self.bert = AutoModel.from_pretrained("./pretrained models/PubmedBERTbase-MimicBig-EntityBERT")
        self.classification = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)
        self.tokenizer = tokenizer
        self.loss = nn.CrossEntropyLoss()

    def forward(self, tokens, labels=None):
        embeddings = self.bert(**tokens)
        embeddings = embeddings.last_hidden_state
        result = []
        batched_result = []
        truth = []
        for sample in range(len(embeddings)):
            # result.append([])
            batched_result.append([])
            # truth.append([])
            for token in range(len(embeddings[sample])):
                token_embedding = embeddings[sample][token]
                logits = self.classification(token_embedding)
                # result[sample].append(logits)
                batched_result[sample].append(logits)
                result.append(logits)
                # truth[sample].append(1 if labels[sample][token] else 0)
                if labels:
                    truth.append(1 if labels[sample][token] else 0)
            # result[sample] = torch.stack(result[sample], dim=0)
            batched_result[sample] = self.softmax(torch.stack(batched_result[sample], dim=0))
        batched_result = torch.stack(batched_result, dim=0)
        result = torch.stack(result, dim=0)
        result = self.softmax(result)
        if labels:
            truth = torch.tensor(truth, device=result.device)
            loss = self.loss(result, truth)
            return {"loss": loss, "results": result, "batched_result": batched_result, "truth": truth, "mask": tokens["attention_mask"]}
        else:
            return {"results": result, "batched_result": batched_result, "truth": truth, "mask": tokens["attention_mask"]}
