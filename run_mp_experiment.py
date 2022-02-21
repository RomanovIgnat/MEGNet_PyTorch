import torch.cuda

from MPDataset import MPDataset
from model.Struct2Graph import SimpleCrystalConverter, GaussianDistanceConverter
from torch_geometric.loader import DataLoader
import click
from model.MEGNet import MEGNet
import torch.nn.functional as F


@click.command()
@click.argument('dataset_path')
def main(dataset_path):
    dataset = MPDataset(dataset_path,
                        pre_transform=SimpleCrystalConverter(bond_converter=GaussianDistanceConverter()))

    dataloader = DataLoader(dataset[5200:5301], batch_size=1, shuffle=False)

    for i, batch in enumerate(dataloader):
        print(i, batch)

    '''
    trainset = dataset[:60000]
    testset = dataset[60000:]

    print(len(trainset))
    print(len(testset))

    trainloader = DataLoader(trainset, batch_size=100, shuffle=True)
    testloader = DataLoader(testset, batch_size=200, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MEGNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(500):

        model.train(True)
        for i, batch in enumerate(trainloader):
            print(batch)
            batch = batch.to(device)
            y = batch.y

            preds = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
            ).squeeze()

            loss = F.l1_loss(preds, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            if not i % 60:
                print(f'{loss.to("cpu").data.numpy(): .3f}', end=" ")

        total = []
        model.train(False)
        with torch.no_grad():
            for batch in testloader:
                batch = batch.to(device)
                y = batch.y

                preds = model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                ).squeeze()

                total.append(F.l1_loss(preds, y, reduction='sum').to('cpu').data.numpy())

            print(sum(total) / 9239)
            '''


if __name__ == '__main__':
    main()
