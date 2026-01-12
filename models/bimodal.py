import torch
from torch import nn

from models.knowledge_graph_encoder import GraphEncoder
from models.text_encoder import EntityBERTtextEncoder


class MultiModalPrediction(nn.Module):
    def __init__(self, number_of_relations=3, combine_embeddings=True):
        super(MultiModalPrediction, self).__init__()
        self.combine_embeddings = combine_embeddings
        self.number_of_relations = number_of_relations
        self.text_embedding_size = 768
        self.reduced_text_embedding_size = 64
        self.graph_embedding_size = 50
        self.graph_model = GraphEncoder(number_of_relations=number_of_relations, output_size=self.graph_embedding_size)
        self.text_model = EntityBERTtextEncoder(number_of_relations=number_of_relations, pooling_strategy='both_events')
        self.pooling_strategy = self.text_model.pooling_strategy
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        if self.combine_embeddings:
            # self.linear = nn.Linear(self.text_embedding_size * 2 + self.graph_embedding_size * 2, self.number_of_relations).double()
            self.linear = nn.Linear(self.reduced_text_embedding_size + self.graph_embedding_size * 2, self.number_of_relations).double()
        else:
            self.linear = nn.Linear(self.number_of_relations * 2, self.number_of_relations).double()

    def forward(self, data, labels):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data.to(device)
        graph_prediction = self.graph_model(data, labels=labels, return_embedding=True)
        text_prediction = self.text_model(data, labels=labels, return_embedding=True)
        concatenated = torch.cat((graph_prediction, text_prediction), 1)
        concatenated = concatenated.double()
        x = self.linear(concatenated)
        # x = self.softmax(x)
        loss = self.criterion(x, labels)
        return {"loss": loss, "predictions": x}
    pass
