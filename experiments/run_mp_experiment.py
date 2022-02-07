from MPDataset import MPDataset
from model.Struct2Graph import SimpleCrystalConverter, GaussianDistanceConverter
from torch_geometric.loader import DataLoader
import click


@click.command()
@click.argument('dataset_path')
def main(dataset_path):
    dataset = MPDataset(dataset_path,
                        pre_transform=SimpleCrystalConverter(bond_converter=GaussianDistanceConverter()))
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    for batch in dataloader:
        print(batch)


if __name__ == '__main__':
    main()
