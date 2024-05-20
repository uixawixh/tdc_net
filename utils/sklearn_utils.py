from typing import TypeVar, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

_T = TypeVar('_T')


class StandardizeFeature(object):
    """
    PyTorch Dataset Transform for standardizing features using scikit-learn's StandardScaler.
    """

    def __init__(self, scaler: StandardScaler):
        self.scaler = scaler

    def __call__(self, sample: _T) -> _T:
        if isinstance(sample, torch.Tensor):
            dtype, device = sample.dtype, sample.device
            sample = sample.cpu().numpy()
            if len(sample.shape) == 1:
                sample = sample.reshape(1, -1)
            sample = self.scaler.transform(sample).reshape(-1)
            sample = torch.tensor(sample, dtype=dtype, device=device)
        else:
            if len(sample.shape) == 1:
                sample = sample.reshape(1, -1)
            sample = self.scaler.transform(sample).reshape(-1)
        return sample


def get_scaler_for_dataset(dataset) -> StandardScaler:
    """
    Fit a scaler on the features of the provided dataset.
    """
    features, _ = get_all_dataset(dataset)
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def get_all_dataset(dataset) -> Tuple[np.ndarray, np.ndarray]:
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    datas = next(iter(dataloader))

    if len(datas) == 3:
        _, feature, target = datas
    else:
        feature, target = datas
    if isinstance(feature, torch.Tensor):
        feature = feature.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    return feature, target
