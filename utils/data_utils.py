# !/usr/bin/python
# coding:utf-8
import os
import random
from functools import reduce
from typing import List, Callable, Dict, Tuple

import torch
import joblib
import numpy as np
from tqdm import tqdm
from ase.db import connect
from ase.db.core import AtomsRow
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from utils.sklearn_utils import StandardizeFeature, get_scaler_for_dataset
from utils.feature_utils import read_structure_file, FeatureExtract
from config import SEED, SCALE


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


class MlpDataset(Dataset):

    def __init__(self, data):
        self.data = data

        # Standardize
        scaler = StandardScaler()
        scaler.fit([i[0] for i in data])
        sf = StandardizeFeature(scaler)
        self.data = [[sf(features), label] for features, label in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        features, target = self.data[item]
        return features, target


class CnnDataset(Dataset):

    def __init__(self, data, extra=None, transform: Callable = None, scaler=None):
        assert extra is None or len(data) == len(extra)
        self.data = [(torch.tensor(i), torch.tensor(j)) for i, j in data]
        self.extra_features = np.asarray(extra, dtype=np.float32)

        self.transform = transform
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(self.extra_features)
        self.scaler = scaler
        if scaler:
            sf = StandardizeFeature(scaler)
            self.extra_features = [torch.from_numpy(sf(item)) for item in self.extra_features]  # Standardize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feature, target = self.data[item]
        extra_features = self.extra_features[item]

        # Data augment
        if self.transform is not None:
            feature = self.transform(feature)

        return feature.float(), extra_features.float(), target.float()


def get_dataloader(
        db_path: str,
        select: Dict,
        target: List[str] | str,
        extra_features: List = None,
        extra_columns: List = None,
        save_path: str = '',
        batch_size: int = 32,
        train_val_test_ratio=(8, 2, 0),
        target_size=(96, 96),
        augment: bool = False,
        drop_col: List[str] = None,
):
    # save == '' indicates no save
    if extra_columns is None:
        extra_columns = extra_features
    assert len(train_val_test_ratio) == 3
    train_loader_path = f'{save_path}/train_loader.pth'
    val_loader_path = f'{save_path}/val_loader.pth'
    test_loader_path = f'{save_path}/test_loader.pth'

    torch.manual_seed(SEED)
    random.seed(SEED)

    picture_size = (target_size[0] * 2, target_size[1] * 2)
    transform = RandomCropAndRotate3D(picture_size) if augment else None
    if any(not os.path.exists(path) for path in [train_loader_path, val_loader_path, test_loader_path]):
        # Get data and shuffle
        data, extra = get_data_from_db(db_path, select, target, *(extra_features or []),
                                       max_size=target_size[0] * target_size[1])
        # Generate features
        fx = FeatureExtract(dir_path=save_path)
        df = fx.get_features(
            [item[0] for item in data],
            [item[1] for item in data],
            data_extra=extra,
            extra_columns=extra_columns,
            drop_col=drop_col
        )

        data = [[i, j] for i, j in zip(df['STRUCTURE'].to_numpy(), df['LABEL'].to_numpy())]

        extra = df.iloc[:, :-2].to_numpy()

        # Calculate ratio of train, validation and test
        total_count = len(data)
        train_count = int(total_count * train_val_test_ratio[0] / sum(train_val_test_ratio))
        val_test_count = total_count - train_count
        val_count = int(val_test_count * train_val_test_ratio[1] / (train_val_test_ratio[1] + train_val_test_ratio[2]))

        # Stratified sampling
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_test_count / total_count, random_state=SEED)
        y = [i[1] for i in data]
        num_bins = np.linspace(start=min(y) - 1e-3, stop=max(y) + 1e-3, num=4)
        y_binned = np.digitize(y, bins=num_bins)
        train_indices, val_test_indices = next(sss.split([i[0] for i in data], y_binned))
        val_indices = val_test_indices[:val_count]
        test_indices = val_test_indices[val_count:]

        train_data = [data[i] for i in train_indices]
        train_extra = [extra[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        val_extra = [extra[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        test_extra = [extra[i] for i in test_indices]

        # Scaler for train dataset only
        train_dataset = CnnDataset(train_data, train_extra, transform=transform)
        scaler = train_dataset.scaler
        save_path and joblib.dump(scaler, f'{save_path}/scaler.joblib')
        val_dataset = CnnDataset(val_data, val_extra, scaler=scaler)
        test_dataset = CnnDataset(test_data, test_extra, scaler=scaler)

    # Load or save the dataset
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


def get_data_from_db(
        db: str,
        select: Dict,
        target: List[str] | str,
        *args,
        max_size,
        target_range=(0, 8),
) -> Tuple[List, List]:
    db = connect(db)
    rows = db.select(**select)
    res = []
    extra_features = []
    for row in tqdm(rows, 'Loading data', unit='row'):
        row: AtomsRow

        atoms = row.toatoms()
        structure = read_structure_file(atoms)

        if structure.lattice.a * structure.lattice.b * np.abs(np.cos(structure.lattice.gamma)) * SCALE ** 2 > max_size:
            # Drop the too big lattice
            continue

        try:
            t = reduce(lambda x, y: x[y], target, row.data) if isinstance(target, list) else getattr(row, target)
            if t is None or t > target_range[1] or t < target_range[0]:
                continue

            tmp = []
            for extra in args:
                e = reduce(lambda x, y: x[y], extra, row.data) if isinstance(extra, list) else getattr(row, extra)
                tmp.append(e)
            extra_features.append(tmp)
            res.append([structure, t])
        except (KeyError, AttributeError):
            continue

    print(f'共计获取到 {len(res)} 条数据')
    return res, extra_features


if __name__ == '__main__':
    from plot_utils import plot_atoms

    get_data_from_db('../datasets/c2db.db', {'selection': 'dos_at_ef_soc'},
                     'dos_at_ef_soc', max_size=96 ** 2)
