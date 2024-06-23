# !/usr/bin/python
# coding:utf-8
from typing import Tuple

import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score, mean_squared_error

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
