# !/usr/bin/python
# coding:utf-8
import os
import time
import shutil
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter

from config import DEVICE
from utils.plot_utils import plot_losses
from utils.eval_utils import evaluate_loss


def save_checkpoint(epoch, train_loss, val_loss, filename: str, model=None, optimizer=None, scheduler=None, **kwargs):
    save_dict = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        **kwargs
    }
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    if model is not None:
        save_dict['model_state_dict'] = model.state_dict()
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(save_dict, filename)


def load_checkpoint(filepath: str, model=None, optimizer=None, scheduler=None, inplace: bool = False):
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)

        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']

        if inplace:
            assert model is not None and optimizer is not None
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(
                f"Checkpoint successfully loaded, the model has been restored to the state at the end of {epoch} round"
            )

            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(DEVICE)

        return epoch, train_loss, val_loss
    else:
        print(f"Checkpoint file not found: '{filepath}'")
        return None, None, None


class Logger:

    def __init__(self, log_dir=''):
        self.log_dir = f'../logs/{log_dir}'
        self.writer = None

    def __enter__(self):
        self.writer = SummaryWriter(self.log_dir)
        return self.writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()
        if exc_type is not None:
            print(f"Error encountered: {exc_val}")
        # Return False to propagate exception, if any
        return False


def checkpoints_losses_plot(
        checkpoint_path: str,
        num_epochs: int,
        *,
        start_epoch: int = 1,
        remove_model: bool = False,
):
    # remove_model == True, then remove the info(model,optimizer,scheduler)
    # but the last checkpoint will be saved
    train_losses, val_losses = [], []
    for idx in range(start_epoch, num_epochs + 1):
        path = os.path.join(checkpoint_path, f'model_epoch_{idx}.ckpt')
        epoch, train_loss, val_loss = load_checkpoint(path)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if idx != num_epochs and remove_model:
            # A safe method, save the new one then remove the previous
            copy_path = f'{path}.copy'
            save_checkpoint(epoch, train_loss, val_loss, copy_path)
            os.remove(path)
            shutil.move(copy_path, path)

    plot_losses(train_losses, val_losses)


def train_and_eval(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        checkpoint_path: str = '',
        start_epoch: int = 1,
        num_epochs: int = 30,
        *,
        classify_boundaries: Tuple = (2, 4, 6)
):
    assert start_epoch >= 1
    train_losses, val_losses = [], []
    classify_boundaries = torch.tensor(classify_boundaries, device=DEVICE, dtype=torch.float)
    for epoch in range(1, start_epoch):
        path = os.path.join(checkpoint_path, f'model_epoch_{epoch}.ckpt')
        _, train_loss, val_loss = load_checkpoint(path, model, optimizer, scheduler, inplace=epoch == start_epoch - 1)
        if train_loss is None:
            print(f'Reset start_epoch to {epoch}')
            start_epoch = epoch
            break
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    model = model.to(DEVICE)
    length = len(train_loader) // 10

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0
        start_time = time.time()
        for i, datas in enumerate(train_loader, 1):
            if len(datas) == 3:
                features, extra, labels = datas
                features, extra, labels = features.to(DEVICE), extra.to(DEVICE), labels.to(DEVICE)
                outputs = model(features, extra)
            else:
                features, labels = datas
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                outputs = model(features)

            if isinstance(criterion, (list, tuple)):
                # The first should be loss of classification
                criterion_classification, criterion_regression = criterion
                predicted_logits, outputs = outputs
                true_classes = torch.bucketize(labels, boundaries=classify_boundaries, right=True)
                classification_loss = criterion_classification(predicted_logits.float(), true_classes)

                regression_loss = criterion_regression(outputs, labels)

                loss = classification_loss + 0.1 * regression_loss
            else:
                loss = criterion(outputs.squeeze(), labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % length == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}')
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_time = time.time() - start_time
        val_loss, r2, mae = evaluate_loss(model, val_loader, criterion, classify_boundaries, (1, 0.01))
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f'Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.4f},',
            f'Validation Loss: {val_loss:.4f}, cost: {train_time:.2f}s\n',
            f'Validation R2: {r2:.2f}, MAE: {mae:.4f}'
        )

        scheduler and scheduler.step()

        checkpoint_path and save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            val_loss=val_loss,
            train_loss=train_loss,
            train_time=train_time,
            scheduler=scheduler,
            filename=f'{checkpoint_path}/model_epoch_{epoch}.ckpt'
        )
    # print(train_losses, val_losses)
    plot_losses(train_losses, val_losses)


if __name__ == '__main__':
    checkpoints_losses_plot('../checkpoints/tdcnet_v1', 60, remove_model=True)
