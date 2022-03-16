from pymatgen.core import Structure, Lattice
from pymatgen.optimization.neighbors import find_points_in_spheres
import numpy as np
from torch_geometric.data import Data
import torch


class SimpleCrystalConverter:
    def __init__(
            self,
            atom_converter=None,
            bond_converter=None,
            cutoff=5.0
    ):
        self.cutoff = cutoff
        self.atom_converter = atom_converter if atom_converter else DummyConverter()
        self.bond_converter = bond_converter if bond_converter else DummyConverter()

    def convert(self, d):
        lattice_matrix = np.ascontiguousarray(np.array(d.lattice.matrix), dtype=float)
        pbc = np.array([1, 1, 1], dtype=int)
        cart_coords = np.ascontiguousarray(np.array(d.cart_coords), dtype=float)

        center_indices, neighbor_indices, _, distances = find_points_in_spheres(
            cart_coords, cart_coords, r=self.cutoff, pbc=pbc, lattice=lattice_matrix, tol=1e-8
        )
        exclude_self = center_indices != neighbor_indices

        edge_index = torch.Tensor(np.stack((center_indices[exclude_self], neighbor_indices[exclude_self]))).long()
        if torch.numel(edge_index) == 0:
            return None

        x = torch.Tensor(self.atom_converter.convert(np.array([i.specie.Z for i in d]))).long()
        edge_attr = torch.Tensor(self.bond_converter.convert(distances[exclude_self]))
        state = getattr(d, "state", None) or [[0.0, 0.0]]
        y = d.y if hasattr(d, "y") else 0
        bond_batch = torch.Tensor([0 for _ in range(edge_index.shape[1])]).long()

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


if __name__ == '__main__':
    structure = Structure(Lattice.cubic(3.167), ['Mo', 'Mo'], [[0, 0, 0], [0.5, 0.5, 0.5]])
    Sconv = SimpleCrystalConverter(bond_converter=GaussianDistanceConverter())
    print(Sconv.convert(structure))
