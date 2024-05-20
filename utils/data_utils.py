# !/usr/bin/python
# coding:utf-8
import os
import math
import json
import random
from functools import reduce
from typing import Union, List, Callable, Dict, Tuple

import torch
import joblib
import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.db import connect
from ase.db.core import AtomsRow
from ase.geometry import get_distances
from pymatgen.core import Structure, Lattice, Element
from pymatgen.io.ase import AseAtomsAdaptor
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split

from config import DEVICE
from utils.sklearn_utils import StandardizeFeature, get_scaler_for_dataset
from config import SEED


class RandomCropAndRotate3D:
    def __init__(self, output_size, flip_prob=0.5, rotate_prob=0.5):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob

    def __call__(self, x):
        # x is expected to be a Tensor of shape (C, D, H, W)
        d, h, w = x.shape[1:]
        new_h, new_w = self.output_size

        # Random crop
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        x = x[:, :, top: top + new_h, left: left + new_w]

        # Random flip
        if random.random() < self.flip_prob:
            x = torch.flip(x, dims=[-1])  # Flip along width

        # Random rotation
        if random.random() < self.rotate_prob:
            x = torch.rot90(x, k=1, dims=[-2, -1])  # Rotate 90 degrees along H-W dimensions

        return x


def get_average_properties(structure: Structure) -> Tuple[float, float, float]:
    electron_affinities = []
    ionization_energies = []
    covalent_radii = []

    for species in structure.species:
        element = Element(species.symbol)
        electron_affinities.append(element.electron_affinity)
        ionization_energies.append(element.ionization_energies[1] if len(element.ionization_energies) > 1 else 0)
        covalent_radii.append(element.atomic_radius)

    avg_electron_affinity = sum(electron_affinities) / len(electron_affinities)
    avg_ionization_energy = sum(ionization_energies) / len(ionization_energies)
    avg_covalent_radius = sum(covalent_radii) / len(covalent_radii)

    return avg_electron_affinity, avg_ionization_energy, avg_covalent_radius


def single_column_descriptor(structure: Structure) -> np.ndarray:
    avg_electron_affinity, avg_ionization_energy, avg_covalent_radius = get_average_properties(structure)
    features = np.array([
        structure.density,
        structure.composition.average_electroneg,
        avg_electron_affinity,
        avg_ionization_energy,
        avg_covalent_radius,
        structure.composition.total_electrons,
        structure.composition.num_atoms,
        structure.get_space_group_info()[1],
    ], dtype=np.float32)
    return features


class MlpDataset(Dataset):

    def __init__(self, data):
        self.data = []
        for structure, target in data:
            features = torch.tensor(single_column_descriptor(structure), dtype=torch.float, device=DEVICE)
            target = torch.tensor(target, dtype=torch.float, device=DEVICE)
            self.data.append((features, target))

        scaler = get_scaler_for_dataset(self)
        sf = StandardizeFeature(scaler)
        self.data = [(sf(item), t) for item, t in self.data]  # Standardize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        features, target = self.data[item]
        if hasattr(self, 'sf'):
            features = self.sf(features)
        return features, target


class CnnDataset(Dataset):

    def __init__(self, data, picture_size, transform: Callable = None, scaler=None):
        self.data = []
        self.extra_features = []
        # Prepare features, because generation is expensive
        for structure, target in tqdm(data, desc='Processing structures', unit='structure'):
            feature = structure_to_feature_v1(structure, n=5, picture_size=picture_size)  # transform to picture
            target = torch.tensor(target)
            self.data.append((feature, target))
            self.extra_features.append(
                torch.tensor(single_column_descriptor(structure))
            )

        self.transform = transform
        if scaler is None:
            scaler = get_scaler_for_dataset(self)
        self.scaler = scaler
        if scaler:
            sf = StandardizeFeature(scaler)
            self.extra_features = [sf(item) for item in self.extra_features]  # Standardize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feature, target = self.data[item]
        extra_features = self.extra_features[item]

        # Data augmentation
        if self.transform is not None:
            feature = self.transform(feature)
        return feature.float(), extra_features.float(), target.float()


def get_dataloader_v1(
        db_path: str,
        select: Dict,
        target: List[str],
        save_path: str = '',
        batch_size: int = 32,
        train_val_test_ratio=(8, 2, 0),
        target_size=(96, 96),
):
    # save == '' indicates no save
    torch.manual_seed(SEED)
    random.seed(SEED)
    assert len(train_val_test_ratio) == 3
    train_loader_path = f'{save_path}/train_loader.pth'
    val_loader_path = f'{save_path}/val_loader.pth'
    test_loader_path = f'{save_path}/test_loader.pth'

    picture_size = (target_size[0] * 2, target_size[1] * 2)
    transform = RandomCropAndRotate3D(picture_size)
    if any(not os.path.exists(path) for path in [train_loader_path, val_loader_path, test_loader_path]):
        data = get_data_from_db(db_path, select, target)
        random.shuffle(data)

        total_count = len(data)
        train_count = int(total_count * train_val_test_ratio[0] / sum(train_val_test_ratio))
        val_test_count = total_count - train_count
        val_count = int(val_test_count * train_val_test_ratio[1] / (train_val_test_ratio[1] + train_val_test_ratio[2]))

        # Scaler for train dataset only
        train_dataset = CnnDataset(data[:train_count], picture_size, transform=transform)
        scaler = train_dataset.scaler
        save_path and joblib.dump(scaler, f'{save_path}/scaler.joblib')
        val_dataset = CnnDataset(data[train_count:train_count + val_count], picture_size, scaler=scaler)
        test_dataset = CnnDataset(data[train_count + val_count:], picture_size, scaler=scaler)

    if os.path.exists(train_loader_path):
        train_loader = torch.load(train_loader_path)
        print('Load train data successfully!')
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        save_path and torch.save(train_loader, train_loader_path)
    if os.path.exists(val_loader_path):
        val_loader = torch.load(val_loader_path)
        print('Load validation data successfully!')
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        save_path and torch.save(val_loader, val_loader_path)
    if os.path.exists(test_loader_path):
        test_loader = torch.load(test_loader_path)
        print('Load test data successfully!')
    else:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        save_path and torch.save(test_loader, test_loader_path)

    return train_loader, val_loader, test_loader


def read_structure_file(filename: Union[str, Atoms]) -> Structure:
    """
    Read file
    :param filename: POSCAR | cif | c2db.json | ase.Atoms
    :return:
    """
    structure = None
    if isinstance(filename, str):
        if filename.endswith('.json'):
            dc = json.loads(filename)
            lst1 = dc['cell']['array']['__ndarray__']
            cell = np.array(lst1[2], dtype=np.float64).reshape(lst1[0])
            lst2 = dc['numbers']['__ndarray__']
            elements = np.array(lst2[2], dtype=np.int32).reshape(lst2[0])
            lst3 = dc['positions']['__ndarray__']
            positions = np.array(lst3[2], dtype=np.float64).reshape(lst3[0])
            structure = Structure(lattice=cell, species=elements, coords=positions)
        else:
            structure = Structure.from_file(filename)
    elif isinstance(filename, Atoms):
        structure = AseAtomsAdaptor.get_structure(atoms=filename)

    assert structure is not None, 'Format not valid'
    return structure


def search_px_for_atom(x: float, y: float, radius: float, lattice_x: float, lattice_y: float, gamma: float,
                       picture_size: tuple, super_cell: bool = True):
    """ complexity: O(r^2) super_cell=True complexity -> O(width * height) """
    height, width = picture_size
    x, y, radius, lattice_x = int(x), int(y), int(radius), int(lattice_x)
    origin = [[], []]
    lattice_y1, lattice_y2 = int(lattice_y * radians_cos(gamma)), int(lattice_y * radians_sin(gamma))
    offset_y = (height - lattice_y2) >> 1
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if np.sqrt(np.square(i - x) + np.square(j - y)) > radius:
                continue

            cross_product_l = np.cross(np.array([i, j]), np.array([lattice_y1, lattice_y2]))
            cross_product_r = np.cross(np.array([i - lattice_x, j]), np.array([lattice_y1, lattice_y2]))
            if cross_product_l < 0:  # point on the left side of the l-line
                if j < 0:
                    new_i = i + lattice_x + lattice_y1
                    new_j = j + lattice_y2
                elif j > lattice_y2:
                    new_i = i + lattice_x - lattice_y1
                    new_j = j - lattice_y2
                else:
                    new_i = i + lattice_x
                    new_j = j
            elif cross_product_r > 0:  # point on the right side of the r-line
                if j < 0:
                    new_i = i - lattice_x + lattice_y1
                    new_j = j + lattice_y2
                elif j > lattice_y2:
                    new_i = i - lattice_x - lattice_y1
                    new_j = j - lattice_y2
                else:
                    new_i = i - lattice_x
                    new_j = j
            else:  # middle
                if j < 0:
                    new_i = i + lattice_y1
                    new_j = j + lattice_y2
                elif j > lattice_y2:
                    new_i = i - lattice_y1
                    new_j = j - lattice_y2
                else:
                    new_i = i
                    new_j = j
            if gamma > 90:
                new_i += abs(lattice_y1)
            origin[0].append(new_i % width)
            origin[1].append(new_j % height)
    if not super_cell:
        return origin

    res = [[], []]

    extend_y = math.ceil(height / lattice_y2)
    if lattice_y1 < 0:
        extend_x = math.ceil((width - extend_y * lattice_y1) / lattice_x)
    else:
        extend_x = math.ceil(width / lattice_x)

    for origin_x, origin_y in zip(*origin):
        for i in range(lattice_y1 // lattice_x, extend_x):
            for j in range(0, extend_y):
                new_x = origin_x + i * lattice_x + j * lattice_y1
                new_y = origin_y + j * lattice_y2
                if 0 <= new_x < width and 0 <= new_y < height:
                    res[0].append(new_x)
                    res[1].append(new_y)

    return res


def radians_cos(x):
    return np.cos(np.radians(x))


def radians_sin(x):
    return np.sin(np.radians(x))


def structure_to_feature_v1(
        structure: Union[str, Structure, Atoms],
        n: int = 5,
        *,
        tolerance: float = 0.1,
        picture_size: tuple = (96, 96),
        is_train: bool = True,
        real_scale: bool = True,
) -> torch.Tensor:
    """
    extract features for CNN
    v1: Divide the two-dimensional material into n layers based on the vacuum layer axis.
        Each atomic layer is filled with one layer.

    :param structure
    :param n: layers
    If it is insufficient, the insufficient layer will be filled with zero.
    If it is exceeded, the middle layer will be evenly pressed into one layer.
    :param tolerance: if |c1 - c2| <= tolerance(unit Ang), then the atoms are considered to be in the same layer
    :param picture_size: height and width of picture
    :param is_train
    :param real_scale:
     false: single-layer filling takes the method of enlarging the lattice until it just fills the picture.
     true: 1px == 0.1Ang.
    :return Tensor(3, n, width, height)
    """
    assert n > 0, 'Layers must be positive'
    if isinstance(structure, (str, Atoms)):
        structure = read_structure_file(structure)
    structure: Structure
    height, width = picture_size

    lattice = structure.lattice
    if all(abs(lattice.matrix[0][i]) > 1e-3 for i in (1, 0)):
        # rotate a parallel to +x
        a = lattice.matrix[0]
        angle_with_x_axis = np.arctan2(a[1], a[0])
        rotation_matrix_z = np.array([
            [np.cos(-angle_with_x_axis), -np.sin(-angle_with_x_axis), 0],
            [np.sin(-angle_with_x_axis), np.cos(-angle_with_x_axis), 0],
            [0, 0, 1]
        ])
        rotated_matrix = np.dot(rotation_matrix_z, lattice.matrix.T).T
        if rotated_matrix[0][0] < 0:
            rotated_matrix[:1, :2] *= -1
        lattice = Lattice(rotated_matrix)  # reset lattice newly
        structure = Structure(lattice, structure.species, structure.frac_coords)

    if real_scale:
        scale = 10
    else:
        # Prepare for enlarging the lattice, equal proportion
        scale_x = width / (lattice.a + np.abs(lattice.b * radians_cos(lattice.gamma)))
        scale_y = height / (lattice.b * radians_sin(lattice.gamma))
        scale = min(scale_x, scale_y)

    # Get all layers, complexity O[atoms * (radius * scale)^2], max(radius * scale) == 224
    species, positions = structure.species, structure.cart_coords
    sorted_symbols_positions = sorted(zip(species, positions), key=lambda z_item: z_item[1][2])
    lattice_x, lattice_y = lattice.a * scale, lattice.b * scale
    unmerged_list: list[list] = []
    prev_z = -10
    for idx, (symbol, positions) in enumerate(sorted_symbols_positions):
        x, y, z = positions[0] * scale, positions[1] * scale, positions[2]

        element = Element(symbol)
        electronegativity = 0 if element.X == np.nan else element.X
        radius, period, group = element.atomic_radius * scale, element.row, element.group
        # make the main clan and the secondary clan distinct
        if 3 <= group <= 12:
            group += 6
        elif 12 <= group <= 17:
            group -= 9
        elif group == 18:
            group = 8
        # Add the atoms, scale the px to range(0,1)
        pending = search_px_for_atom(x, y, radius, lattice_x, lattice_y, lattice.gamma, picture_size)
        cur = [
            [electronegativity / 4, pending[0], pending[1]],
            [period / 6, pending[0], pending[1]],
            [group / 18, pending[0], pending[1]],
        ]
        if z - prev_z > tolerance:
            unmerged_list.append([cur])
        else:
            unmerged_list[-1].append(cur)
        prev_z = z

    # find the merged indices
    layers = len(unmerged_list)
    left = right = layers >> 1
    merge_num = max(layers - n + 1, 0)
    if merge_num > 0:
        tmp = merge_num >> 1
        left -= tmp
        right += merge_num - tmp

    return_matrix = np.zeros((3, n, height, width), dtype=np.float64)
    # fill the picture
    for idx, item in enumerate(unmerged_list):
        if left <= idx < right:
            for e, p, g in item:
                return_matrix[0, left, e[1], e[2]] += e[0]
                return_matrix[1, left, p[1], p[2]] += p[0]
                return_matrix[2, left, g[1], g[2]] += g[0]
        else:
            index = idx if idx < merge_num else idx - merge_num + 1
            for e, p, g in item:
                return_matrix[0, index, e[1], e[2]] = e[0]
                return_matrix[1, index, p[1], p[2]] = p[0]
                return_matrix[2, index, g[1], g[2]] = g[0]
    if merge_num > 1:
        return_matrix[left] /= merge_num

    return torch.tensor(return_matrix, requires_grad=is_train)


def get_data_from_db(db: str, select: Dict, target: List[str], target_range=(0, 8)) -> List:
    db = connect(db)
    rows = db.select(**select)
    res = []
    for row in tqdm(rows, 'Loading data', unit='row'):
        row: AtomsRow

        atoms = row.toatoms()
        structure = read_structure_file(atoms)
        try:
            t = reduce(lambda x, y: x[y], target, row.data)
            if t is None or t > target_range[1] or t < target_range[0]:
                continue
            res.append([structure, t])
        except KeyError:
            # print(f'{structure.formula}: none')
            continue
    print(f'共计获取到 {len(res)} 条数据')
    return res


def calc_rdf(structure, dr=0.1, rmax=10.0):
    import matplotlib.pyplot as plt
    bins = np.arange(0, rmax + dr, dr)

    if isinstance(structure, Structure):
        structure = structure.to_ase_atoms()
    d, d_len = get_distances(structure.positions, cell=structure.cell, pbc=True)
    hist, bins = np.histogram(d_len[d_len > 0], bins=bins)

    volume = structure.get_volume()
    density = len(structure) / volume
    rdf = hist / (4 * np.pi * (bins[:-1] ** 2) * dr * density * len(structure) + 1e-10)
    rdf.astype(np.float32)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(bins[:-1], rdf)
    plt.xlabel('Distance [Å]')
    plt.ylabel('g(r)')
    plt.show()

    return rdf


if __name__ == '__main__':
    from plot_utils import plot_atoms

    get_data_from_db('../datasets/c2db.db', select={},
                     target=['results-asr.gs.json', 'kwargs', 'data', 'gap_nosoc'])
