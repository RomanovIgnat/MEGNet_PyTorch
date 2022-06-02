import random
import click
import os.path as osp
import numpy as np

from megnet.models import MEGNetModel
from MEGNet.utils.utils import set_random_seed
from MEGNet.utils.utils import String2StructConverter
from monty.serialization import loadfn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph


class MEGNetOriginalTrainer:
    def __init__(self, trainset, testset, config):
        pass

    def train(self):
        pass


@click.command()
@click.argument('dataset_path')
def main(dataset_path):
    set_random_seed(17)

    raw_data = loadfn(osp.join(dataset_path, "mp.2018.6.1.json"))
    random.shuffle(raw_data)
    train_data = raw_data[:64500]
    test_data = raw_data[64500:]

    converter = String2StructConverter('formation_energy_per_atom')

    train_data = [converter.convert(s) for s in tqdm(train_data)]
    test_data = [converter.convert(s) for s in tqdm(test_data)]

    train_target = np.array([s.y for s in train_data])
    test_target = np.array([s.y for s in test_data])

    cg = CrystalGraph(bond_converter=GaussianDistance(np.linspace(0, 5, 100), 0.5), cutoff=4)
    model = MEGNetModel(100, 2, 16, graph_converter=cg)

    model.train(train_data, list(train_target), epochs=2)

    preds = model.predict_structures(test_data)
    print(mean_absolute_error(test_target, preds))


if __name__ == "__main__":
    main()
