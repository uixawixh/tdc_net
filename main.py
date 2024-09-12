# !/usr/bin/python
# coding:utf-8
import os
import sys
import pathlib
import argparse

import config

parser = argparse.ArgumentParser(description='Two-Dimensional Crystal Neural Networks')

parser.add_argument('dataset_dir_path', metavar='OPTIONS',
                    help='Directory of the dataset to be trained')

parser.add_argument('-t', '--task', choices=['regression', 'classification', 'features'],
                    default='regression', help='Complete a regression or classification task (default: regression)')

parser.add_argument('-m', '--model', choices=['xgboost', 'tdcnet'], default='tdcnet',
                    help='The model for your task (default: tdcnet)')

parser.add_argument('--no-load-data', action='store_true',
                    help='Load the data you saved. You should use it if you want to do another split.')
parser.add_argument('--train-ratio', default=0.8, type=float,
                    help='number of training data to be loaded (default 0.8)')
parser.add_argument('--val-ratio', default=0.2, type=float,
                    help='percentage of validation data to be loaded (default 0.2)')
parser.add_argument('--test-ratio', default=0, type=float,
                    help='percentage of test data to be loaded (default 0)')

features_generator_group = parser.add_argument_group('Features generator')
features_generator_group.add_argument('--corr-criterion', default=0.8, type=float,
                                      help='One of the columns with a Pearson correlation coefficient'
                                           ' exceeding this flag will be deleted (default: 0.8)')

hyperparam_group = parser.add_argument_group('TDCNet Hyperparameters')
hyperparam_group.add_argument('--epochs', default=100, type=int,
                              help='number of total epochs to run (default: 100)')

hyperparam_group.add_argument('-b', '--batch-size', default=64, type=int,
                              help='batch size (default: 64)')

hyperparam_group.add_argument('--no-drop-last', action='store_true',
                              help='Do not drop the last batch if its size < batch size')

hyperparam_group.add_argument('-lr', '--learning-rate', default=0.001, type=float)

hyperparam_group.add_argument('--weight-decay', default=4e-3, type=float)

hyperparam_group.add_argument('--step-size', default=50, type=int,
                              help='Step size of the learning rate scheduler')

hyperparam_group.add_argument('--step-gamma', default=0.6, type=float,
                              help='Step gamma of the learning rate scheduler')

hyperparam_group.add_argument('--augment', action='store_true',
                              help='Picture data augment')

parser.add_argument('--no-save', action='store_true',
                    help='Do not save the model param, scaler param and the split result')

parser.add_argument('-ft', '--crystal-file-type', default='cif', choices=['cif', 'vasp'])

args = parser.parse_args(sys.argv[1:])


def main():
    model_name = args.model
    task = args.task

    if task == 'regression' and model_name == 'tdcnet':
        train_tdc_net()


def generate_features_from_structures(structures):
    from utils.feature_utils import FeatureExtract

    fe = FeatureExtract(dir_path=None if args.no_save else args.dataset_dir_path)
    fe.get_features(

    )


def train_tdc_net(id_target_csv: str = 'id_prop.csv'):
    import torch
    import numpy as np
    import pandas as pd
    from torch import nn
    from pymatgen.core import Structure

    from models.tdc_net import simple_net
    from models.base_model import initialize_weights
    from utils.training_utils import train_and_eval
    from utils.data_utils import get_dataloader

    global args

    dir_path = args.dataset_dir_path
    path = pathlib.Path(dir_path)
    if not path.exists() or not path.is_dir():
        raise NotADirectoryError(f'Check the {dir_path}!')
    ckpt_dir = path / 'checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    csv_path = pathlib.Path(path / id_target_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f'You need a file named {id_target_csv}')
    df = pd.read_csv(csv_path, dtype=np.object_)
    n_extra_features = max(df.shape[1] - 2, 0)

    structures, extra_features, labels = [], None, None
    for _id, row in zip(df.iloc[:, 0].to_numpy(dtype=np.str_), df.iloc[:, 1:].to_numpy(dtype=np.float32)):
        if '.' in _id:
            filename = _id
        else:
            filename = f'{_id}.{args.crystal_file_type}'

        structures.append(Structure.from_file(path / filename))

        if n_extra_features == 0:
            if labels is None:
                labels = row.reshape(1, -1)
            else:
                labels = np.vstack([labels, row.reshape(1, -1)])
        else:
            if extra_features is None:
                extra_features = row[:-1].reshape(1, -1)
                labels = row[-1:].reshape(1, -1)
            else:
                extra_features = np.vstack([extra_features, row[:-1].reshape(1, -1)])
                labels = np.vstack([labels, row[-1:].reshape(1, -1)])

    datasets_dir = path / 'datasets'
    os.makedirs(datasets_dir, exist_ok=True)
    train_loader, val_loader, test_loader = get_dataloader(
        structures,
        labels,
        extra_features,
        extra_columns=df.iloc[:, 1:-1].columns if n_extra_features > 0 else None,
        scaler_path=datasets_dir,
        save_path=None if args.no_save else datasets_dir,
        batch_size=args.batch_size,
        train_val_test_ratio=(args.train_ratio, args.val_ratio, args.test_ratio),
        augment=args.augment,
        load_data=not args.no_load_data
    )
    structure_feature, tabular_features, _ = next(iter(train_loader))

    model = simple_net(
        num_features=tabular_features.shape[1],
        in_channels=structure_feature.shape[1],
    )
    initialize_weights(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.step_gamma)
    if args.task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()

    train_and_eval(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        checkpoint_path=ckpt_dir.name,
        checkpoint_step=1,
        start_epoch=1,
        num_epochs=args.epochs,
    )


if __name__ == '__main__':
    main()
