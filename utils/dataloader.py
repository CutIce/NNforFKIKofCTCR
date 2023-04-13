import torch.utils.data as Data
import random
import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

# class CTCRDataset(Dataset):
#
#     def __init__(self, path="./dataset/CRL-Dataset-CTCR-Pose.csv", version="train"):
#         assert version in ["train", "valid", "test"]
#         self.version = version
#         self.data = self.__load_data__()
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return len(self.data)




def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)

