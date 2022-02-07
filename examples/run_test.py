from monty.serialization import loadfn
from MEGNet_PyTorch.model.MEGNet import MEGNet
import torch
from torch_geometric.data import Batch
from MEGNet_PyTorch.model.Struct2Graph import SimpleCrystalConverter, GaussianDistanceConverter
import torch.nn.functional as F


if __name__ == '__main__':
    data = loadfn("./bulk_moduli.json")
    structures = data["structures"]
    target = torch.log10_(torch.Tensor(data["bulk_moduli"]))

    converter = SimpleCrystalConverter(bond_converter=GaussianDistanceConverter())

    structures_converted = [converter.convert(s) for s in structures]
    for s in structures_converted:
        s.bond_batch = torch.Tensor([0 for _ in range(s.edge_index.shape[1])]).long()

    batch = Batch.from_data_list(structures_converted)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    M = MEGNet().to(device)
    opt = torch.optim.Adam(M.parameters(), lr=1e-3)

    batch = batch.to(device)
    target = target.to(device)

    M.train(True)
    for i in range(1000):
        preds = M(batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch).squeeze()

        loss = F.mse_loss(preds, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print("Loss", loss.data.numpy())

