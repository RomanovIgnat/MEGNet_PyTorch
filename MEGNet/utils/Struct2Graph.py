import torch
import numpy as np
from torch_geometric.data import Data
from pymatgen.core import Structure
from pymatgen.core.periodic_table import DummySpecies
from pymatgen.optimization.neighbors import find_points_in_spheres


class MyTensor(torch.Tensor):
    """
    this class is needed to work with graphs without edges
    """
    def max(self, *args, **kwargs):
        if torch.numel(self) == 0:
            return 0
        else:
            return torch.max(self, *args, **kwargs)


class SimpleCrystalConverter:
    def __init__(
            self,
            target_name,
            atom_converter=None,
            bond_converter=None,
            add_z_bond_coord=False,
            cutoff=5.0
    ):
        """
        Parameters
        ----------
        atom_converter: converter that converts pymatgen structure to node features
        bond_converter: converter that converts distances to edge features
        add_z_bond_coord: use z-coordinate feature or no
        cutoff: cutoff radius
        """
        self.target_name = target_name
        self.cutoff = cutoff
        self.atom_converter = atom_converter if atom_converter else DummyConverter()
        self.bond_converter = bond_converter if bond_converter else DummyConverter()
        self.add_z_bond_coord = add_z_bond_coord

    def convert(self, d):
        lattice_matrix = np.ascontiguousarray(np.array(d.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
        cart_coords = np.ascontiguousarray(np.array(d.cart_coords), dtype=float)

        center_indices, neighbor_indices, _, distances = find_points_in_spheres(
            cart_coords, cart_coords, r=self.cutoff, pbc=pbc, lattice=lattice_matrix, tol=1e-8
        )

        exclude_self = (center_indices != neighbor_indices)

        edge_index = torch.Tensor(np.stack((center_indices[exclude_self], neighbor_indices[exclude_self]))).long()

        x = torch.Tensor(self.atom_converter.convert(d)).long()

        distances_preprocessed = distances[exclude_self]
        if self.add_z_bond_coord:
            z_coord_diff = np.abs(cart_coords[edge_index[0], 2] - cart_coords[edge_index[1], 2])
            distances_preprocessed = np.stack(
                (distances_preprocessed, z_coord_diff), axis=0
            )

        edge_attr = torch.Tensor(self.bond_converter.convert(distances_preprocessed))
        state = getattr(d, "state", None) or [[0.0, 0.0]]
        y = getattr(d, self.target_name) if hasattr(d, self.target_name) else 0
        bond_batch = MyTensor(np.zeros(edge_index.shape[1])).long()

        return Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr, state=torch.Tensor(state), y=y, bond_batch=bond_batch
        )

    def __call__(self, d):
        return self.convert(d)


class DummyConverter:
    def convert(self, d):
        return d.reshape((-1, 1))


class GaussianDistanceConverter:
    def __init__(self, centers=np.linspace(0, 5, 100), sigma=0.5):
        self.centers = centers
        self.sigma = sigma

    def convert(self, d):
        return np.exp(
            -((d.reshape((-1, 1)) - self.centers.reshape((1, -1))) / self.sigma) ** 2
        )

    def get_shape(self):
        return len(self.centers)


class FlattenGaussianDistanceConverter(GaussianDistanceConverter):
    def __init__(self, centers=np.linspace(0, 5, 100), sigma=0.5):
        super().__init__(centers, sigma)

    def convert(self, d):
        res = []
        for arr in d:
            res.append(super().convert(arr))
        return np.hstack(res)

    def get_shape(self):
        return 2 * len(self.centers)


class AtomFeaturesExtractor:
    def __init__(self, atom_features):
        self.atom_features = atom_features

    def convert(self, structure: Structure):
        if self.atom_features == "Z":
            return np.array(
                [0 if isinstance(i, DummySpecies) else i.Z for i in structure.species]
            ).reshape(-1, 1)
        elif self.atom_features == 'werespecies':
            return np.array([
                [
                    0 if isinstance(i, DummySpecies) else i.Z,
                    i.properties["was"],
                ] for i in structure.species
            ])
        else:
            return np.array(
                [0 if isinstance(i, DummySpecies) else i.Z for i in structure.species]
            ).reshape(-1, 1)

    def get_shape(self):
        if self.atom_features == "Z":
            return 1
        elif self.atom_features == 'werespecies':
            return 2
        else:
            return None
