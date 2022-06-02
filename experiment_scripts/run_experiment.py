import random

import click
import numpy as np
import yaml

from monty.serialization import loadfn
from tqdm import tqdm

from MEGNet.utils.utils import set_random_seed
from MEGNet.utils.utils import String2StructConverter
from MEGNet.MEGNet_Trainer import MEGNetTrainer


def set_return(o, name, val):
    setattr(o, name, val)
    return o


def parse_data(path, is_str):
    print("reading data")
    raw_data = loadfn(path)
    if is_str:
        converter = String2StructConverter(['formation_energy_per_atom', 'band_gap'])
        structures_list = [converter.convert(s) for s in tqdm(raw_data)]
    else:
        structures_list = raw_data["structures"]
        targets = np.log10(raw_data["bulk_moduli"])
        structures_list = [set_return(s, "bulk_moduli", float(t)) for s, t in tqdm(zip(structures_list, targets))]
    return structures_list


@click.command()
@click.argument('dataset_path')
@click.argument('experiment_config_path')
@click.argument('model_config_path')
def main(dataset_path, experiment_config_path, model_config_path):
    set_random_seed(17)

    with open(experiment_config_path) as ey:
        experiment_config = yaml.safe_load(ey)
    with open(model_config_path) as my:
        model_config = yaml.safe_load(my)

    dataset = parse_data(dataset_path, experiment_config['is_str'])
    if experiment_config['shuffle']:
        random.shuffle(dataset)

    test_size = int(len(dataset) * experiment_config['test_size'])
    trainset = dataset[:-test_size]
    testset = dataset[-test_size:]
    print(f"len of trainset: {len(trainset)}")
    print(f"len of testset: {len(testset)}")

    if experiment_config['trainer'] == 'pytorch_trainer':
        trainer = MEGNetTrainer(trainset, testset, model_config)
    else:
        raise NotImplementedError

    trainer.train()


if __name__ == '__main__':
    main()
