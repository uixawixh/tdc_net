# !/usr/bin/python
# coding:utf-8
from typing import Tuple

import joblib
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.feature_utils import structure_to_feature
from models.tdc_net import WeightedMSELoss
from config import DEVICE


def evaluate_loss(
        model,
        dataloader,
        criterion,
) -> Tuple[float, float, float, float]:
    if next(model.parameters()).device != DEVICE:
        model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
        for datas in dataloader:
            features = datas[:-1]
            features = [i.to(DEVICE) for i in features] if isinstance(features, (tuple, list)) else features.to(DEVICE)
            labels = datas[-1].to(DEVICE)
            outputs = model(*features)
            pbe_gaps = features[1][:, -1]

            if isinstance(criterion, WeightedMSELoss):
                loss = criterion(outputs.squeeze(), labels, pbe_gaps)
            else:
                loss = criterion(outputs.squeeze(), labels)

            total_loss += loss.item()

            outputs = outputs.view(-1)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(outputs.cpu().numpy())

    # Concatenate all batches
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    # Calculate the average loss
    avg_loss = total_loss / len(dataloader)

    # Calculate R2 and MAE
    r2 = r2_score(all_labels, all_predictions)
    mae = mean_absolute_error(all_labels, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))

    return avg_loss, r2, mae, rmse


def predict_value(model, dataloader) -> Tuple[np.ndarray, np.ndarray]:
    model.to(DEVICE)
    model.eval()

    true_values = []
    pred_values = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data[:-1]
            labels = data[-1]
            if isinstance(inputs, (list, tuple)):
                for idx, item in enumerate(inputs):
                    inputs[idx] = item.to(DEVICE)
                outputs = model(*inputs)
            else:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)

            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            true_values.append(labels.flatten())
            pred_values.append(outputs.flatten())

    true_values = np.hstack(true_values)
    pred_values = np.hstack(pred_values)
    return true_values, pred_values


def evaluate_data(model, single_features, structures=None):
    model.to(DEVICE)
    model.eval()

    pred_values = []
    single_features = torch.tensor(single_features).to(DEVICE)

    outputs = []
    length = single_features.shape[0]
    if structures is None:
        idx = 0
        while idx < length:
            end = min(idx + 64, length)
            outputs.extend(model(single_features[idx:end].cpu().detach().numpy()))
            idx = end
    else:
        idx = 0
        while idx < length:
            end = min(idx + 64, length)
            single = single_features[idx:end]
            st = torch.tensor(np.array([
                structure_to_feature(structure) for structure in structures[idx:end]
            ])).to(DEVICE)
            outputs.extend(model(st, single).cpu().detach().numpy())
            idx = end

    outputs = np.array(outputs)
    pred_values.append(outputs.flatten())

    pred_values = np.hstack(pred_values)
    return pred_values


def evaluate(model, dataloader) -> np.ndarray:
    model.to(DEVICE)
    model.eval()

    pred_values = []

    with torch.no_grad():
        for data in dataloader:
            inputs = data[:-1]
            if isinstance(inputs, (list, tuple)):
                for idx, item in enumerate(inputs):
                    inputs[idx] = item.to(DEVICE)
                outputs = model(*inputs)
            else:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)

            outputs = outputs.cpu().numpy()
            pred_values.append(outputs.flatten())

    pred_values = np.hstack(pred_values)
    return pred_values


if __name__ == '__main__':
    import pandas as pd
    import torch.nn.functional as F
    from torch import nn
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler

    from utils.training_utils import load_checkpoint
    from utils.data_utils import get_dataloader, get_data_from_db
    from utils.plot_utils import plot_corr_scatter, plot_true_predict
    from models.tdc_net import simple_net
    from utils.feature_utils import FeatureExtract

    df = pd.read_csv('../datasets/hse_set_pbe/train_data.zip')
    columns = list(df.columns)
    columns.remove('LABEL')
    columns.remove('gap')

    select = {
        'selection': 'gap',
        'filter': lambda row: getattr(
            row, 'hform', 1) <= 0 and getattr(
            row, 'ehull', 1) <= 0.05 and getattr(
            row, 'magmom', 0) != 0 and getattr(
            row, 'dynamic_stability_stiffness', False) and getattr(
            row, 'dynamic_stability_phonons', False) and getattr(
            row, 'natoms') < 10
        # and abs(row.gap_dir - row.gap) <= 1e-5
    }

    train_loader, val_loader, test_loader = get_dataloader(
        '../datasets/c2db.db',
        save_path='../datasets/test',
        batch_size=64,
        select=select,
        target='gap',
        train_val_test_ratio=(0, 0, 10),
        select_col=columns,
        base_path='../datasets/hse_set_pbe',
        extra_features=['gap']
    )

    # classify
    df2 = pd.read_csv('../datasets/hse_set_pbe/train_data.zip')
    X, y = df2.iloc[:, :-1].to_numpy(), (df2['LABEL'] <= 1e-4).to_numpy()
    model = make_pipeline(
        MinMaxScaler(),
        XGBClassifier(random_state=59, max_depth=3, n_estimators=200)
    )
    model.fit(X, y)
    is_metals = model.predict(df.iloc[:, :-1].to_numpy())

    # regression
    model_path = '../datasets/pbe_set/best_model.ckpt'
    model = simple_net(
        num_features=74,
        prior_func=lambda x: F.relu6(x) * 8 / 6,
        in_channels=8,
    )

    criterion = nn.MSELoss()

    load_checkpoint(model_path, model, inplace=True)

    from utils.data_utils import get_data_from_db

    data, _ = get_data_from_db(
        '../datasets/c2db.db',
        select=select,
        target='gap',
        max_size=96 ** 2
    )
    structures, gaps = [item[0] for item in data], [item[1] for item in data]

    res = []
    ss = []
    fe = FeatureExtract('../models/dir')
    for s in structures:
        ss.append(s)

    features = fe.get_features(
        ss, select_col=columns, with_label=False
    ).to_numpy().astype(np.float32)
    X = features
    scaler = joblib.load('../datasets/pbe_set/scaler.joblib')
    X = scaler.transform(X)

    evals = evaluate(model, test_loader)
    gap_hse_structure = []
    for idx, structure in enumerate(structures):
        gap_hse, gap, is_metal = evals[idx], df['gap'][idx], is_metals[idx]
        if not is_metal:
            gap_hse_structure.append((gap, gap_hse, structure))

    print(len(gap_hse_structure))
    plot_corr_scatter([item[0] for item in gap_hse_structure], [item[1] for item in gap_hse_structure], 10000)
    plot_true_predict([item[0] for item in gap_hse_structure], [item[1] for item in gap_hse_structure])
