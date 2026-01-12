import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

class GraphLanguageModel(nn.Module):
    def __init__(self, number_of_relations=3):
        super(GraphLanguageModel, self).__init__()
        self.device = "cpu"
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.number_of_relations = number_of_relations
        modelcard = 'plenz/GLM-t5-small'
        model_output_size = 256
        self.model = AutoModel.from_pretrained(modelcard, trust_remote_code=True, revision='main')
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(modelcard)
        self.mode = "global"  # global or local

        self.linear = nn.Linear(model_output_size * 2, self.number_of_relations).double()
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data, labels):
        # data.to(self.device)

        inputs = []
        for i in range(len(data["text_relations"])):
            graph = self.model.data_processor.encode_graph(
                tokenizer=self.tokenizer, g=data["text_relations"][i],
                             text=data["text"][i], how=self.mode)
            inputs.append(graph)
        model_inputs = self.model.data_processor.to_batch(
            data_instances=inputs, tokenizer=self.tokenizer, max_seq_len=None, device=self.device)
        outputs = self.model(**model_inputs)
        model_output = []
        for i in range(len(data["text_relations"])):
            event1 = data["text"][i][data["event1_start"][i]:data["event1_end"][i]]
            event2 = data["text"][i][data["event2_start"][i]:data["event2_end"][i]]
            embedding1 = self.model.data_processor.get_embedding(sequence_embedding=outputs.last_hidden_state[i],
                                                            indices=inputs[i].indices, concept=event1,
                                                            embedding_aggregation='mean')
            embedding2 = self.model.data_processor.get_embedding(sequence_embedding=outputs.last_hidden_state[i],
                                                            indices=inputs[i].indices, concept=event2,
                                                            embedding_aggregation='mean')
            output = torch.cat((embedding1, embedding2))
            model_output.append(output)

        x = self.linear(torch.stack(model_output))
        x = self.softmax(x)
        loss = self.criterion(x, labels)
        return {"loss": loss, "logits": x}