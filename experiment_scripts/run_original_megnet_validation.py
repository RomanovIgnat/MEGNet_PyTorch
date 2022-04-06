import random
import click
import os.path as osp
import numpy as np

from megnet.utils.models import load_model
from MEGNet.utils.utils import set_random_seed
from MEGNet.utils.utils import String2StructConverter
from monty.serialization import loadfn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error


@click.command()
@click.argument('dataset_path')
def main(dataset_path):
    set_random_seed(17)

    raw_data = loadfn(osp.join(dataset_path, "mp.2018.6.1.json"))
    random.shuffle(raw_data)
    raw_data = raw_data[:4585]
    converter = String2StructConverter('formation_energy_per_atom')
    structures_list = [converter.convert(s) for s in tqdm(raw_data)]
    target_array = np.array([s.y for s in structures_list])

    model = load_model("Eform_MP_2018")
    preds = model.predict_structures(structures_list)
    print(mean_absolute_error(target_array, preds.squeeze()))


if __name__ == "__main__":
    main()
