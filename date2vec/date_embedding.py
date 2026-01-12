import os

import date2vec.Model
from date2vec.Model import Date2VecConvert
import torch

# Date2Vec embedder object
# Loads a pretrained model
d2v = Date2VecConvert(model_path="./date2vec/pretrained/date2vec.pth")

def compute_date_embedding(year, month, day, hours=0, minutes=0, seconds=0, size=768):
    x = torch.Tensor([[int(hours), int(minutes), int(seconds), int(year), int(month), int(day)]]).float()
    embed = d2v(x)
    vector = torch.cat((embed.reshape(-1), torch.zeros(size - 64)))
    return vector

# if __name__ == '__main__':
#     a = torch.load("./pretrained/d2v_98291_17.169918439404636.pth", map_location=torch.device('cpu'))
#     print(a)
#     b = date2vec.Model.Date2Vec(k=64, act="cos")
#     b.load_state_dict(state_dict=a.state_dict())
#     print(b)
#     torch.save(b, "./pretrained/date2vec.pth")
