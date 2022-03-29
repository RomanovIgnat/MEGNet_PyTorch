import numpy as np
from copy import copy


class Scaler:
    def __init__(self):
        self.mean = 0
        self.std = 1.0

    def fit(self, dataset, feature_name='y'):
        data = np.array([getattr(dataset.get(i), feature_name).data.numpy() for i in range(len(dataset))])
        self.mean = np.mean(data)
        self.std = np.std(data)

    def transform(self, data):
        data_copy = copy(data)
        return (data_copy - self.mean) / (self.std if abs(self.std) > 1e-7 else 1.)

    def inverse_transform(self, data):
        data_copy = copy(data)
        std = self.std if abs(self.std) > 1e-7 else 1.0
        return data_copy * std + self.mean
