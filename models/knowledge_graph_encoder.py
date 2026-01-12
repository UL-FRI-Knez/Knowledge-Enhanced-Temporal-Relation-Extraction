import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

import torch_geometric.nn as pyg_nn

class TemporalRelationAggregation(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels, edge_features):
        super(TemporalRelationAggregation, self).__init__(aggr='add')  # "Add" aggregation.
        self.layers = 2
        self.out_channels = out_channels
        self.edge_features = edge_features
        self.lin = nn.Linear(in_channels + edge_features, out_channels, bias=True)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, edge_attr):
        # Transform node feature matrix.
        self_x = self.lin_self(x)
        return self_x + self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # _i central node that collects information
        # _j neighbour node
        if len(x_j) > 0 and len(edge_attr) > 0:
            return self.lin(torch.cat((x_j, edge_attr), 1))
        else:
            return torch.zeros(0, self.out_channels)


class GraphEncoder(nn.Module):
    def __init__(self, node_size=768, edge_size=768, output_size=50, number_of_relations=3, dropout=0.2):
        super(GraphEncoder, self).__init__()
        self.node_size = node_size
        self.edge_size = edge_size
        self.number_of_relations = number_of_relations
        self.pooling_strategy = "concat_entities"
        self.hidden_dim = output_size

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)

        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(self.node_size, self.edge_size, self.hidden_dim))
        self.lns = nn.ModuleList()
        self.num_layers = 2
        for l in range(self.num_layers - 1):
            self.lns.append(nn.LayerNorm(self.hidden_dim).float())
            self.convs.append(self.build_conv_model(self.hidden_dim, self.edge_size, self.hidden_dim))
        self.linear = nn.Linear(self.hidden_dim, self.hidden_dim).float()
        self.post_mp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, number_of_relations))
        self.dropout = dropout
        pass
    def build_conv_model(self, node_size, edge_size, output_node_size):
        return TemporalRelationAggregation(node_size, output_node_size, edge_size)

    def forward(self, data, labels, return_embedding=False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data.to(device)
        # if not self.training:
        #     return nn.functional.one_hot(data.rule_based_prediction).double()
        x, edge_index, batch, event1, event2, edge_attr = data.x, data.edge_index, data.batch, data.event1_index, data.event2_index, data.edge_attr

        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x=x.float(), edge_index=edge_index.type(torch.int64), edge_attr=edge_attr.float())
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = self.linear(x)

        event1_emb = x[event1.int()]
        event2_emb = x[event2.int()]
        x = torch.cat((event1_emb, event2_emb), 1)

        if return_embedding:
            return x

        x = self.post_mp(x)
        x = self.softmax(x)
        loss = self.criterion(x, labels)
        return {"loss": loss, "logits": x}
