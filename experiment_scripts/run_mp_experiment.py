import torch.cuda
import click
import torch.nn.functional as F

from MEGNet.utils.MPDataset import MPDataset
from MEGNet.utils.Struct2Graph import SimpleCrystalConverter, GaussianDistanceConverter
from torch_geometric.loader import DataLoader
from MEGNet.model.MEGNet import MEGNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from MEGNet.utils.utils import Scaler
from MEGNet.utils.utils import set_random_seed


@click.command()
@click.argument('dataset_path')
def main(dataset_path):
    dataset = MPDataset(dataset_path,
                        pre_transform=SimpleCrystalConverter(bond_converter=GaussianDistanceConverter()))
    set_random_seed(17)
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
    scheduler = ReduceLROnPlateau(optimizer=opt, factor=0.5, patience=150, threshold=5e-2, verbose=True, min_lr=1e-4)

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
