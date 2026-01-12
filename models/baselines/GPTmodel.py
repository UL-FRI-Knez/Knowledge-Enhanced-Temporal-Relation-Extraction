import torch
from torch import nn

from graph_building.llm.OpenChat import request_open_Chat


class GPTTemporalRelationExtraction(nn.Module):
    def __init__(self, plm_model='bert-base-cased', number_of_relations=3, dropout=0.2, deeper_network=False, pooling_strategy='both_events'):
        super(GPTTemporalRelationExtraction, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, data, labels):
        text, event1_start, event1_end, event2_start, event2_end = data.text, data.event1_start, data.event1_end, data.event2_start, data.event2_end
        results = []
        for i in range(len(text)):
            prompt = "Input document D = " + text[i]
            prompt += ("\n Given the document D and a list of temporal relations [before, after, vague, equal] and event " +
                       "triggers that are labeled as <e1></e1> and <e2></e2>. what is the temporal relation between " +
                       text[i][event1_start[i] - 4: event1_end[i] + 5] +
                       " and " +
                       text[i][event2_start[i] - 4: event2_end[i] + 5] +
                       "? Answer vague if unsure. Keep the answer short and concise")
            answer = request_open_Chat(prompt)
            if "before" in answer.lower():
                results.append([1.0, 0.0, 0.0])
                continue
            if "after" in answer.lower():
                results.append([0.0,1.0,0.0])
                continue
            results.append([0.0,0.0,1.0])
        x = torch.tensor(results).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        loss = self.criterion(x, labels)
        return {"logits": x, "loss": loss}