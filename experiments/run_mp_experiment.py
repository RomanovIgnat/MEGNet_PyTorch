from MPDataset import MPDataset
from MEGNet_PyTorch.model.Struct2Graph import SimpleCrystalConverter, GaussianDistanceConverter
from torch_geometric.loader import DataLoader


dataset = MPDataset("./mp.2018.dataset", pre_transform=SimpleCrystalConverter(bond_converter=GaussianDistanceConverter()))
dataloader = DataLoader(dataset, batch_size=100, shuffle=False)


for batch in dataloader:
    print(batch)
    print(batch.y)
    print(batch.bond_batch)
