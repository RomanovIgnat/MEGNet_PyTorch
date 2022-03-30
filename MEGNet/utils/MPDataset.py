import torch
from torch_geometric.data import InMemoryDataset
from monty.serialization import loadfn
import os.path as osp
from tqdm import tqdm
from .utils import String2StructConverter


class MPDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["mp.2018.6.1.json"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        raw_data = loadfn(osp.join(self.raw_dir, "mp.2018.6.1.json"))

        converter = String2StructConverter('formation_energy_per_atom')
        structures_list = [converter.convert(s) for s in tqdm(raw_data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in structures_list]
            data_list = [data for data in data_list if data]
        else:
            raise "you should give struct2graph converter"

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
