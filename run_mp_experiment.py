import torch.cuda

from MPDataset import MPDataset
from model.Struct2Graph import SimpleCrystalConverter, GaussianDistanceConverter
from torch_geometric.loader import DataLoader
import click
from model.MEGNet import MEGNet
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import Scaler


@click.command()
@click.argument('dataset_path')
def main(dataset_path):
    dataset = MPDataset(dataset_path,
                        pre_transform=SimpleCrystalConverter(bond_converter=GaussianDistanceConverter()))
    torch.manual_seed(17)
    dataset = dataset.shuffle()

    trainset = dataset[:64500]
    testset = dataset[64500:]

    scaler = Scaler()
    scaler.fit(trainset)

    print(len(trainset))
    print(len(testset))

    trainloader = DataLoader(trainset, batch_size=200, shuffle=True)
    testloader = DataLoader(testset, batch_size=200, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MEGNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer=opt, factor=0.5, patience=100, threshold=5e-2, verbose=True, min_lr=1e-4)

    for epoch in range(2000):

        print(epoch, end=" ")

        model.train(True)
        for i, batch in enumerate(trainloader):
            batch = batch.to(device)
            y = scaler.transform(batch.y)

            preds = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
            ).squeeze()

            loss = F.mse_loss(y, preds)
            loss.backward()
            opt.step()
            opt.zero_grad()

            if not i % 32:
                print(f'{1000 * loss.to("cpu").data.numpy(): .3f}', end=" ")

        total = []
        model.train(False)
        with torch.no_grad():
            for batch in testloader:
                batch = batch.to(device)
                y = batch.y

                preds = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                ).squeeze()

                total.append(F.l1_loss(scaler.inverse_transform(preds), y, reduction='sum').to('cpu').data.numpy())

            print(sum(total) / len(testset))

        scheduler.step(sum(total) / len(testset))


if __name__ == '__main__':
    main()
