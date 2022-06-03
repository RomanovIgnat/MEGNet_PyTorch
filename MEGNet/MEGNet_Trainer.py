import numpy as np
import torch.nn.functional as F
import torch

from tqdm import tqdm
from torch_geometric.loader import DataLoader

from .utils.Struct2Graph import (
    SimpleCrystalConverter,
    FlattenGaussianDistanceConverter,
    GaussianDistanceConverter,
    AtomFeaturesExtractor,
)
from .model.MEGNet import MEGNet
from .utils.utils import Scaler


class MEGNetTrainer:
    def __init__(self, trainset, testset, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = config
        self.target_name = config['model']['target_name']

        if self.config["data"]["add_z_bond_coord"]:
            bond_converter = FlattenGaussianDistanceConverter(
                centers=np.linspace(0, self.config['data']['cutoff'], self.config['model']['edge_embed_size'])
            )
        else:
            bond_converter = GaussianDistanceConverter(
                centers=np.linspace(0, self.config['data']['cutoff'], self.config['model']['edge_embed_size'])
            )
        atom_converter = AtomFeaturesExtractor(self.config["data"]["atom_features"])

        self.model = MEGNet(
            edge_input_shape=bond_converter.get_shape(),
            node_input_shape=atom_converter.get_shape(),
            state_input_shape=self.config["model"]["state_input_shape"]
        ).to(self.device)
        self.Scaler = Scaler()

        self.converter = SimpleCrystalConverter(
            target_name=self.config['model']['target_name'],
            bond_converter=bond_converter,
            atom_converter=atom_converter,
            cutoff=self.config["data"]["cutoff"],
            add_z_bond_coord=self.config["data"]["add_z_bond_coord"]
        )
        print("converting data")
        self.train_structures = [self.converter.convert(s) for s in tqdm(trainset)]
        self.test_structures = [self.converter.convert(s) for s in tqdm(testset)]
        self.Scaler.fit(self.train_structures)

        self.trainloader = DataLoader(
            self.train_structures,
            batch_size=self.config["model"]["train_batch_size"],
            shuffle=True,
        )
        self.testloader = DataLoader(
            self.test_structures,
            batch_size=self.config["model"]["test_batch_size"],
            shuffle=False,
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['optim']['lr_initial'])
        if self.config["optim"]["scheduler"].lower() == "ReduceLROnPlateau".lower():
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.config["optim"]["factor"],
                patience=self.config["optim"]["patience"],
                threshold=self.config["optim"]["threshold"],
                min_lr=self.config["optim"]["min_lr"],
                verbose=True,
            )
        else :
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.99
            )

    def one_epoch(self):
        total = []
        self.model.train(True)
        for batch in tqdm(self.trainloader):
            batch = batch.to(self.device)
            y = self.Scaler.transform(batch.y)

            preds = self.model(
                batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
            ).squeeze()

            loss = F.mse_loss(y, preds)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total.append(
                F.l1_loss(self.Scaler.inverse_transform(preds), batch.y, reduction='sum').to('cpu').data.numpy()
            )
        return sum(total) / len(self.train_structures)

    def validation(self):
        total = []
        self.model.train(False)
        with torch.no_grad():
            for batch in self.testloader:
                batch = batch.to(self.device)
                y = batch.y

                preds = self.model(
                    batch.x, batch.edge_index, batch.edge_attr, batch.state, batch.batch, batch.bond_batch
                ).squeeze()

                total.append(F.l1_loss(self.Scaler.inverse_transform(preds), y, reduction='sum').to('cpu').data.numpy())

        return sum(total) / len(self.test_structures)

    def train(self):
        for epoch in range(self.config['model']['epochs']):
            print(f"===={epoch} out of {self.config['model']['epochs'] - 1} epochs====")
            print(f'target: {self.target_name}')

            train_loss = self.one_epoch()
            validation_loss = self.validation()

            self.scheduler.step(train_loss)

            print(f"train loss: {train_loss}, test loss: {validation_loss}")
