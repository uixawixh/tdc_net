import os
import re
import json
import math
from typing import List, Tuple, Optional, Union, Sequence, Any
from itertools import chain

import numpy as np
import pandas as pd
from scipy import stats
from ase import Atoms
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Element, Composition, Lattice

from config import SCALE, SEED

MASS = 0
ATOMIC_RADII = 1
ELECTRON_NEG = 2
IONIZATION_ENERGIES = 3
ELECTRON_AFFINITIES = 4
IONIC_RADII = 5
VALENCE = 6
CORE_CHARGE = 7
SHELL = 8
S_E = 9
P_E = 10
D_E = 11
F_E = 12
BLOCK = 13
S_UNFILLED = 14
P_UNFILLED = 15
D_UNFILLED = 16
F_UNFILLED = 17
MAX_OXIDATION_STATE = 18
MELTING_POINT = 19
MENDELEEV_NO = 20
MOLAR_VOLUME = 21
BOILING_POINT = 22

FEATURES = [
    'MASS',
    'ATOMIC_RADII',
    'ELECTRON_NEG',
    'IONIZATION_ENERGIES',
    'ELECTRON_AFFINITIES',
    'IONIC_RADII',
    'VALENCE',
    'CORE_CHARGE',
    'SHELL',
    'S_E',
    'P_E',
    'D_E',
    'F_E',
    'BLOCK',
    'S_UNFILLED',
    'P_UNFILLED',
    'D_UNFILLED',
    'F_UNFILLED',
    'MAX_OXIDATION_STATE',
    'MELTING_POINT',
    'MENDELEEV_NO',
    'MOLAR_VOLUME',
    'BOILING_POINT',
]
SPDF = ['s', 'p', 'd', 'f']


class FeatureExtract:
    TRAIN = 'train_data'
    TEST = 'test_data'

    def __init__(self, dir_path: str):
        """
        :param dir_path: Read or save file.
        """
        assert os.path.exists(dir_path)
        self.dir_path = dir_path

        self.columns = []

    def get_features(
            self,
            data_structure: Sequence[Structure] = None,
            data_y: Sequence[Any] = None,
            *,
            drop_col: List[str] = None,
            data_extra: Sequence[Sequence[float]] = None,
            extra_columns: List[str] = None,
            save: bool = True,
            picture_feature: bool = True,
    ):
        """
        If there are dataset files, read them
        """
        # Define a function to convert string to np.array
        if extra_columns is None:
            extra_columns = []
        if data_extra is not None:
            assert len(data_extra[0]) == len(extra_columns), 'Must fill the extra_columns for extra features !'
        if drop_col is None:
            drop_col = []

        def get_data(path):
            if os.path.exists(f'{self.dir_path}/{path}.zip'):
                res = pd.read_csv(f'{self.dir_path}/{path}.zip')
                self.columns = res.columns
            elif os.path.exists(f'{self.dir_path}/{path}.csv'):
                res = pd.read_csv(f'{self.dir_path}/{path}.csv')
                self.columns = res.columns
            else:
                self.columns += extra_columns + ['STRUCTURE', 'LABEL']

                res = pd.DataFrame(columns=self.columns)
                if picture_feature:
                    res['STRUCTURE'] = pd.Series([structure_to_feature(x) for x in tqdm(
                        data_structure, "Converting structures to Matrix", total=len(data_structure), unit='row'
                    )])
                res['LABEL'] = np.asarray(data_y)

                extra_features = np.array([single_column_descriptor(x) for x in tqdm(
                    data_structure, "Converting structures to Single", total=len(data_structure), unit='row'
                )], dtype=np.float32)
                if data_extra is not None:
                    extra_features = np.hstack([extra_features, np.array(data_extra)], dtype=np.float32)
                res.iloc[:, :-2] = extra_features

                zero_std_columns = check_zero_std(extra_features, self.columns)
                highly_corr_columns = highly_correlated_columns(extra_features, self.columns)
                index_drop = sorted(
                    set(highly_corr_columns + zero_std_columns + [self.columns.index(i) for i in drop_col]),
                    reverse=True
                )
                res.drop(columns=res.columns[index_drop], inplace=True)
                for i in index_drop:
                    self.columns.pop(i)

            return res

        # Generate all columns of cif features
        for idx, name in enumerate(FEATURES):
            self.columns.append(name + '_MEAN')
            self.columns.append(name + '_STD')
            self.columns.append(name + '_RANGE')
            self.columns.append(name + '_MAX')
            self.columns.append(name + '_MIN')
            self.columns.append(name + '_MODE')
        self.columns += [
            'ELEMENT_NUM',
            'L2_NORM',
            'N_ATOMS',
            'SPACE_GROUP',
            'GAMMA',
            'A',
            'B',
            'C',
            'VOLUME',
            'DENSITY',
        ]

        # Generate train dataset
        train = get_data(self.TRAIN)
        if save:
            self.save(train)

        return train

    def save(self, train: pd.DataFrame, test: Optional[pd.DataFrame] = None, compression: bool = True):
        """ Save datasets, only save the single features """
        train = train.drop(['STRUCTURE'], axis=1, errors='ignore')
        if test is not None:
            test = test.drop(['STRUCTURE'], axis=1, inplace=True, errors='ignore')
        if compression:
            train.to_csv(f'{self.dir_path}/{self.TRAIN}.zip', index=False,
                         compression={'method': 'zip', 'archive_name': f'{self.TRAIN}.csv'})
            if test is not None:
                test.to_csv(f'{self.dir_path}/{self.TEST}.zip', index=False,
                            compression={'method': 'zip', 'archive_name': f'{self.TEST}.csv'})
        else:
            train.to_csv(f'{self.dir_path}/{self.TRAIN}.csv', index=False)
            if test is not None:
                test.to_csv(f'{self.dir_path}/{self.TEST}.csv', index=False)


def highly_correlated_columns(arr: np.ndarray, columns: List, threshold: float = 0.8) -> List:
    import warnings
    # There might be some columns std == 0
    warnings.filterwarnings('ignore')

    corr_matrix = np.abs(np.corrcoef(arr, rowvar=False))

    indices_to_remove = set()

    n = corr_matrix.shape[0]

    for i in range(n):
        if i in indices_to_remove:
            continue
        for j in range(i + 1, n):
            if np.isnan(corr_matrix[i, j]):
                continue
            if corr_matrix[i, j] > threshold:
                indices_to_remove.add(j)
    if len(indices_to_remove) > 0:
        print('-- Warning --')
        print(f'Columns with highly correlated: {[columns[i] for i in indices_to_remove]}')
        print('These features will drop !!!')
        print('-- Warning --\n')

    warnings.filterwarnings('default')

    return list(indices_to_remove)


def check_zero_std(X, columns: List) -> List:
    std_dev = np.std(X, axis=0)
    zero_std_columns = np.where(std_dev < 1e-8)[0]
    if zero_std_columns.size > 0:
        print('-- Warning --')
        print(f'Columns with zero standard deviation: {[columns[i] for i in zero_std_columns]}')
        print('These features will drop !!!')
        print('-- Warning --\n')
    return list(zero_std_columns)


def get_composition_features(composition: Composition | Structure | str) -> np.ndarray:
    """Get all properties from Composition -> len(22 * 6 + 3)"""
    if isinstance(composition, str):
        composition = Composition(composition)
    elif isinstance(composition, Structure):
        composition = composition.composition
    composition: Composition
    overall = [[] for _ in range(len(FEATURES))]

    for element, count in {element: amount for element, amount in composition.items()}.items():
        element: Element
        for _ in range(int(count)):
            overall[MASS].append(element.atomic_mass)
            overall[ATOMIC_RADII].append(element.atomic_radius)
            overall[ELECTRON_NEG].append(element.X)
            overall[IONIZATION_ENERGIES].append(
                element.ionization_energies[1] if len(element.ionization_energies) > 1 else 0)
            overall[ELECTRON_AFFINITIES].append(element.electron_affinity)
            overall[IONIC_RADII].append(element.average_ionic_radius)
            try:
                overall[VALENCE].append(element.valence[1])
            except ValueError:
                overall[VALENCE].append(0)
            overall[CORE_CHARGE].append(element.Z)
            overall[SHELL].append(element.row)
            s = d = p = f = 0
            for _, orbit, num in parse_electronic_structure(element.electronic_structure):
                s += (orbit == 's') * num
                p += (orbit == 'p') * num
                d += (orbit == 'p') * num
                f += (orbit == 'p') * num
            overall[S_E].append(s)
            overall[P_E].append(p)
            overall[D_E].append(d)
            overall[F_E].append(f)
            overall[BLOCK].append(SPDF.index(element.block))
            unfilled = calculate_unfilled_subshells(element)
            overall[S_UNFILLED].append(unfilled[0])
            overall[P_UNFILLED].append(unfilled[1])
            overall[D_UNFILLED].append(unfilled[2])
            overall[F_UNFILLED].append(unfilled[3])
            overall[MAX_OXIDATION_STATE].append(element.max_oxidation_state)
            overall[MELTING_POINT].append(element.melting_point)
            overall[MENDELEEV_NO].append(element.mendeleev_no)
            overall[MOLAR_VOLUME].append(element.molar_volume)
            overall[BOILING_POINT].append(element.boiling_point)

    statistical_features = np.array(
        list(chain.from_iterable(map(lambda item: get_statistical_features(item), overall))), dtype=np.float32
    )

    elements_num = len(composition)
    total_amount = sum(composition.values())
    molar_fractions = [amount / total_amount for amount in composition.values()]
    l2_norm = np.linalg.norm(molar_fractions, 2)
    return np.hstack([statistical_features, [elements_num, l2_norm, composition.num_atoms]], dtype=np.float32)


def parse_electronic_structure(electronic_structure: str) -> List[tuple]:
    electronic_structure = re.sub(r'\[.*?\\]', '', electronic_structure)
    subshells = re.findall(r'(\d+)([spdf])(\d+)', electronic_structure)
    parsed_structure = [(int(n), subshell, int(electrons)) for n, subshell, electrons in subshells]
    return parsed_structure


def calculate_unfilled_subshells(element: Element) -> List:
    electron_configuration = parse_electronic_structure(element.electronic_structure)
    max_electrons = [2, 6, 10, 14]

    unfilled = [0] * 4
    for _, subshell, electrons in electron_configuration:
        index = SPDF.index(subshell)
        unfilled[index] = max(max_electrons[index] - electrons, unfilled[index])

    return unfilled


def get_statistical_features(elements: Sequence[float]) -> Tuple:
    elements_array = np.array(elements, dtype=np.float32)
    mean = np.mean(elements_array)
    std_dev = np.std(elements_array)
    max_val = np.max(elements_array)
    min_val = np.min(elements_array)
    range_val = max_val - min_val
    mode_val = np.average(stats.mode(elements_array)[0])

    return mean, std_dev, range_val, max_val, min_val, mode_val


def get_vacuum_layer_thickness(structure: Structure) -> float:
    lattice = structure.lattice
    c = lattice.c

    z_coords = [site.frac_coords[2] for site in structure.sites]
    max_z = max(z_coords)
    min_z = min(z_coords)

    layer_thickness = (max_z - min_z) * c
    return min(c - layer_thickness, 1)  # Avoid dividing zero


def get_structure_features(structure: Structure) -> np.ndarray:
    a, b = max(structure.lattice.a, structure.lattice.b), min(structure.lattice.a, structure.lattice.b)
    c = structure.lattice.c
    vacuum = get_vacuum_layer_thickness(structure)

    structure_features = np.array(
        [
            structure.get_space_group_info()[1],
            structure.lattice.gamma % 90,
            a,
            b,
            c - vacuum,
            structure.volume / c,
            structure.density * (c - vacuum) / c,
        ]
    )

    # distances = []
    # nn = CrystalNN()
    # for i in range(len(structure)):
    #     nn_info = nn.get_nn_info(structure, i)
    #     for neighbor in nn_info:
    #         site = neighbor['site']
    #         distances.append(structure[neighbor['site_index']].distance(site))
    # structure_features = np.hstack([
    #     structure_features,
    #     get_statistical_features(distances)
    # ], dtype=np.float32)

    return structure_features


def single_column_descriptor(structure: Structure) -> np.ndarray:
    """ Generate single column features -> len(135 + 7) """
    component_features = get_composition_features(structure.composition)
    structure_features = get_structure_features(structure)

    features = np.hstack([
        component_features,
        structure_features
    ], dtype=np.float32)
    return features


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
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if np.sqrt(np.square(i - x) + np.square(j - y)) > radius:
                continue

            cross_product_r = np.cross(np.array([i - lattice_x, j]), np.array([lattice_y1, lattice_y2]))
            cross_product_l = np.cross(np.array([i, j]), np.array([lattice_y1, lattice_y2]))
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


def structure_to_feature(
        structure: Union[str, Structure, Atoms],
        n: int = 5,
        *,
        tolerance: float = 0.1,
        picture_size: tuple = (96, 96),
        real_scale: bool = True,
) -> np.ndarray:
    """
    Extract features for CNN
    Divide the two-dimensional material into n layers based on the vacuum layer axis.
    Each atomic layer is filled with one layer.

    :param structure
    :param n: layers
    If it is insufficient, the insufficient layer will be filled with zero.
    If it is exceeded, the middle layer will be evenly pressed into one layer.
    :param tolerance: if |c1 - c2| <= tolerance(unit Ang), then the atoms are considered to be in the same layer
    :param picture_size: height and width of picture
    :param real_scale:
     false: single-layer filling takes the method of enlarging the lattice until it just fills the picture.
     true: 1px == 0.1Ang.
    :return ndarray(3, n, width, height)
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
        scale = SCALE
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
        e = 0 if element.X == np.nan else element.X
        radius, period, group = element.atomic_radius * scale, element.row, element.group
        # Add the atoms, scale the px to range(0,1)
        pending = search_px_for_atom(x, y, radius, lattice_x, lattice_y, lattice.gamma, picture_size)
        cur = [
            [period * 40, pending[0], pending[1]],
            [group * 15, pending[0], pending[1]],
            [element.Z * 2.5, pending[0], pending[1]],
            [e * 60, pending[0], pending[1]],
            [element.electron_affinity * 0.6, pending[0], pending[1]],
            [element.boiling_point * 0.03, pending[0], pending[1]],
            [element.melting_point * 0.06, pending[0], pending[1]],
            [element.mendeleev_no * 2.2, pending[0], pending[1]],
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

    return_matrix = np.zeros((8, n, height, width), dtype=np.float32)
    # fill the picture
    for idx, item in enumerate(unmerged_list):
        if left <= idx < right:
            for e in item:
                for feature_idx in range(len(e)):
                    return_matrix[feature_idx, left, e[feature_idx][1], e[feature_idx][2]] += e[feature_idx][0]
        else:
            index = idx if idx < merge_num else idx - merge_num + 1
            for e in item:
                for feature_idx in range(len(e)):
                    return_matrix[feature_idx, index, e[feature_idx][1], e[feature_idx][2]] = e[feature_idx][0]
    if merge_num > 1:
        return_matrix[left] /= merge_num

    # Apply Gaussian blur to each layer
    for i in range(n):
        return_matrix[0, i] = return_matrix[0, i]
        return_matrix[1, i] = return_matrix[1, i]
        return_matrix[2, i] = return_matrix[2, i]

    return return_matrix
