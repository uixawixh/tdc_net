# !/usr/bin/python
# coding:utf-8
import os
import sys
import pathlib
import argparse

import joblib
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from pymatgen.core import Structure
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, mean_absolute_error

import config
from utils.feature_utils import FeatureExtract
from models.tdc_net import simple_net
from models.base_model import initialize_weights
from utils.training_utils import train_and_eval
from utils.data_utils import get_dataloader
from utils.plot_utils import plot_shap

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

xgb_hyperparam_group = parser.add_argument_group('XGBoost Hyperparameters')

xgb_hyperparam_group.add_argument('-xlr', '--xgb-learning-rate', default=0.1, type=float,
                                  help='Learning rate (shrinkage factor) to prevent overfitting (default: 0.1)')
xgb_hyperparam_group.add_argument('--n-estimators', default=100, type=int,
                                  help='Number of boosting rounds/trees (default: 100)')
xgb_hyperparam_group.add_argument('--max-depth', default=3, type=int,
                                  help='Maximum depth of a tree (default: 3). Higher depth can lead to overfitting')
xgb_hyperparam_group.add_argument('--min-child-weight', default=3, type=int,
                                  help='Minimum sum of instance weight (hessian) needed in a child (default: 3)')
xgb_hyperparam_group.add_argument('--gamma', default=0.0, type=float,
                                  help='Minimum loss reduction required to make a further partition (default: 0.0)')
xgb_hyperparam_group.add_argument('--subsample', default=1.0, type=float,
                                  help='Subsample ratio of the training instances (default: 1.0, full data)')
xgb_hyperparam_group.add_argument('--colsample-bytree', default=1.0, type=float,
                                  help='Subsample ratio of columns when constructing each tree '
                                       '(default: 1.0, full features)')
xgb_hyperparam_group.add_argument('--reg-lambda', default=1e-4, type=float,
                                  help='L2 regularization term on weights (default: 1e-4). '
                                       'Helps control model complexity')
xgb_hyperparam_group.add_argument('--reg-alpha', default=0.0, type=float,
                                  help='L1 regularization term on weights (default: 0.0). '
                                       'Encourages sparsity')
xgb_hyperparam_group.add_argument('--scale-pos-weight', default=1, type=int,
                                  help='Balancing of positive and negative weights. Used for imbalanced datasets '
                                       '(default: 1)')

xgb_hyperparam_group = parser.add_argument_group('XGBoost Settings')

xgb_hyperparam_group.add_argument('-cv', '--cross-validate', default=5, type=int)
xgb_hyperparam_group.add_argument('--use-grid-search', action='store_true',
                                  help='Use grid search to get best hyperparams')

parser.add_argument('--no-scaler', action='store_true', help='Do not use scaler')

parser.add_argument('--no-save', action='store_true',
                    help='Do not save the model param, scaler param and the split result')

parser.add_argument('-ft', '--crystal-file-type', default='cif', choices=['cif', 'vasp'])

args = parser.parse_args(sys.argv[1:])


def main():
    model_name = args.model
    task = args.task

    if task == 'regression' and model_name == 'tdcnet':
        train_tdc_net()
    elif model_name == 'xgboost':
        train_xgboost()
    else:
        raise


def generate_features_from_structures(structures):
    from utils.feature_utils import FeatureExtract

    fe = FeatureExtract(dir_path=None if args.no_save else args.dataset_dir_path)
    fe.get_features(

    )


def train_tdc_net(id_target_csv: str = 'id_prop.csv'):
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

    structures, extra_features, labels = get_structures_extra_labels(df, path, n_extra_features)
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
        load_data=not args.no_load_data,
        use_scaler=not args.no_scaler,
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
        criterion = nn.NLLLoss()

    train_and_eval(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
        checkpoint_path=ckpt_dir,
        checkpoint_step=1,
        start_epoch=1,
        num_epochs=args.epochs,
    )


def train_xgboost(id_target_csv: str = 'id_prop.csv'):
    global args

    dir_path = args.dataset_dir_path
    path = pathlib.Path(dir_path)
    if not path.exists() or not path.is_dir():
        raise NotADirectoryError(f'Check the {dir_path}!')

    params = {
        'learning_rate': args.xgb_learning_rate,
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'min_child_weight': args.min_child_weight,
        'gamma': args.gamma,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'reg_lambda': args.reg_lambda,
        'reg_alpha': args.reg_alpha,
        'scale_pos_weight': args.scale_pos_weight
    }
    if args.task == 'regression':
        model = XGBRegressor(**params)
    else:
        model = XGBClassifier(**params)

    if not args.no_scaler:
        model = make_pipeline(
            MinMaxScaler(),
            model
        )

    csv_path = pathlib.Path(path / id_target_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f'You need a file named {id_target_csv}')
    df = pd.read_csv(csv_path, dtype=np.object_)
    n_extra_features = max(df.shape[1] - 2, 0)

    fe = FeatureExtract(dir_path)
    structures, extra_features, labels = get_structures_extra_labels(df, path, n_extra_features)
    df_features = fe.get_features(
        structures,
        labels,
        data_extra=extra_features,
        extra_columns=list(df.columns)[1:-1] if n_extra_features > 0 else None,
        save=False,
        picture_feature=False,
        with_label=True,
    )
    feature_names = list(df_features.columns)[:-1]

    X, y = df_features.iloc[:, :-1].to_numpy(), df_features.iloc[:, -1].to_numpy()
    temp_size = args.test_ratio + args.val_ratio
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=temp_size, random_state=config.SEED)
    if args.test_ratio != 0:
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                        test_size=temp_size - args.val_ratio,
                                                        random_state=config.SEED)
        pd.DataFrame(np.hstack([X_test, y_test]), columns=df_features.columns).to_csv(path / 'test.csv')

    pd.DataFrame(np.hstack([X_train, y_train.reshape(-1, 1)]), columns=df_features.columns).to_csv(
        path / 'train.csv')
    pd.DataFrame(np.hstack([X_val, y_val.reshape(-1, 1)]), columns=df_features.columns).to_csv(path / 'val.csv')
    print('The data has been generated!')

    model.fit(X_train, y_train)
    if not args.no_save:
        joblib.dump(model, path / 'xgboost.joblib')

    evaluate_model(model, X_train, y_train, X_val, y_val, args.task)

    plot_xgboost(model, X_train, y_train, X_val, y_val, args.task, feature_names)


def plot_xgboost(model, X_train, y_train, X_val, y_val, task_type, feature_names):
    if isinstance(model, Pipeline):
        scaler = model.named_steps['minmaxscaler']
        X_train, X_val = scaler.transform(X_train), scaler.transform(X_val)
        model = model.named_steps['xgbregressor'] if task_type == 'regression' else model.named_steps['xgbclassfier']

    if task_type == 'regression':
        plot_shap(X_train, model, feature_names)
    else:
        plot_shap(X_train, model, feature_names)


def evaluate_model(model, X_train, y_train, X_test, y_test, task_type):
    if task_type == 'classification':
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        try:
            y_train_pred_prob = model.predict_proba(X_train)[:, 1]
            train_auc = roc_auc_score(y_train, y_train_pred_prob)

            y_test_pred_prob = model.predict_proba(X_test)[:, 1]
            test_auc = roc_auc_score(y_test, y_test_pred_prob)
        except AttributeError:
            train_auc = "N/A (No probability prediction)"
            test_auc = "N/A (No probability prediction)"

        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Train AUC: {train_auc}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test AUC: {test_auc}")

    elif task_type == 'regression':
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)

        y_test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)

        print(f"Train R²: {train_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Train MAE: {train_mae:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test MAE: {test_mae:.4f}")


def get_structures_extra_labels(df, path, n_extra_features):
    structures, extra_features, labels = [], None, None
    for _id, row in tqdm(
            zip(df.iloc[:, 0].to_numpy(dtype=np.str_), df.iloc[:, 1:].to_numpy(dtype=np.float32)),
            desc='Generate features', total=len(df), unit='row'
    ):
        filename = _id if '.' in _id else f'{_id}.{args.crystal_file_type}'
        file_path = path / filename
        if not file_path.exists():
            continue
        structures.append(Structure.from_file(file_path))

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

    return structures, extra_features, labels


if __name__ == '__main__':
    main()
