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
        classify_boundaries=None,
        loss_weight=None
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

            if len(outputs) == 2:
                # Calculate mixed loss
                criterion_classification, criterion_regression = criterion
                predicted_logits, outputs = outputs
                true_classes = torch.bucketize(labels, boundaries=classify_boundaries, right=True)
                classification_loss = criterion_classification(predicted_logits.float(), true_classes)

                regression_loss = criterion_regression(outputs, labels)
                loss = loss_weight[0] * classification_loss + loss_weight[1] * regression_loss
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

    if isinstance(criterion, (list, tuple)):
        # Classification score
        true_classes = np.digitize(all_labels, classify_boundaries.cpu().numpy(), right=True)
        predict_classes = np.digitize(all_predictions, classify_boundaries.cpu().numpy(), right=True)

        accuracy = accuracy_score(true_classes, predict_classes)
        print(f"Accuracy: {accuracy}")
        f1_weighted = f1_score(true_classes, predict_classes, average='weighted')
        print(f"F1 Score (Weighted): {f1_weighted}")

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

            outputs = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)

            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            true_values.append(labels.flatten())
            pred_values.append(outputs.flatten())

    true_values = np.hstack(true_values)
    pred_values = np.hstack(pred_values)
    return true_values, pred_values
