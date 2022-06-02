import numpy as np
from copy import copy
import torch
import random

from pymatgen.io.cif import CifParser


class Scaler:
    def __init__(self):
        self.mean = 0
        self.std = 1.0

    def fit(self, dataset, feature_name='y'):
        data = np.array([getattr(dataset[i], feature_name) for i in range(len(dataset))])
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        data_copy = copy(data)
        return (data_copy - self.mean) / (self.std if abs(self.std) > 1e-7 else 1.)

    def inverse_transform(self, data):
        data_copy = copy(data)
        std = self.std if abs(self.std) > 1e-7 else 1.0
        return data_copy * std + self.mean


class String2StructConverter:
    def __init__(self, struct_target_names):
        self.target_names = struct_target_names

    def convert(self, elem):
        struct = CifParser.from_string(elem['structure']).get_structures()[0]
        for name in self.target_names:
            setattr(struct, name, elem[name])
        return struct


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
