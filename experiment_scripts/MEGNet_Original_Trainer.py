import numpy as np

from megnet.models import MEGNetModel
from megnet.data.graph import GaussianDistance
from megnet.data.crystal import CrystalGraph
from megnet.utils.preprocessing import StandardScaler


class MEGNetOriginalTrainer:
    def __init__(self, trainset, testset, config):
        self.config = config

        self.train_structures = trainset
        self.test_structures = testset
        self.train_targets = [getattr(s, self.config['model']['target_name']) for s in trainset]
        self.test_targets = [getattr(s, self.config['model']['target_name']) for s in testset]

        self.cg = CrystalGraph(
            bond_converter=GaussianDistance(
                np.linspace(0, self.config['data']['cutoff'], self.config['model']['nfeat_edge'])
            ),
            cutoff=self.config['data']['cutoff'],
        )

        self.Scaler = StandardScaler.from_training_data(trainset, self.train_targets)

        self.model = MEGNetModel(
            nfeat_edge=self.config['model']['nfeat_edge'],
            nfeat_global=self.config['model']['nfeat_global'],
            nfeat_node=self.config['model']['nfeat_node'],
            lr=self.config['model']['learning_rate'],
            graph_converter=self.cg,
            target_scaler=self.Scaler,
            metrics=['mae']
        )

    def train(self):
        self.model.train(
            self.train_structures,
            self.train_targets,
            self.test_structures,
            self.test_targets,
            epochs=self.config['model']['epochs'],
            save_checkpoint=False,
            dirname="experiment_scripts"
        )
