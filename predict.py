import sys
import pathlib
import argparse

import joblib
import numpy as np
import pandas as pd

from main import get_structures_extra_labels
from utils.feature_utils import FeatureExtract

parser = argparse.ArgumentParser(description='Prediction model')
parser.add_argument('model_path', help='path to the trained model.')
parser.add_argument('predict_path', help='path to the directory of crystal files.')
parser.add_argument('--csv', action='store_true',
                    help='Use csv file to predict')
parser.add_argument('--has-label', action='store_true',
                    help='The csv file\'s last column is label')

args = parser.parse_args(sys.argv[1:])


def main():
    pass


def predict_tdc_net(id_target_csv: str = 'id_target.csv'):
    pass


def predict_xgboost(id_target_csv: str = 'id_target.csv'):
    global args

    dir_path = args.model_path
    predict_path = pathlib.Path(args.predict_path) / 'train.csv'
    path = pathlib.Path(dir_path)
    if not path.exists() or not path.is_dir():
        raise NotADirectoryError(f'Check the {dir_path}!')
    if not predict_path.exists() or not predict_path.is_file():
        raise NotADirectoryError(f'Check the {predict_path}!')

    model = joblib.load('xgboost.joblib')

    csv_path = pathlib.Path(path / id_target_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f'You need a file named {id_target_csv}')
    df = pd.read_csv(csv_path, dtype=np.object_)
    n_extra_features = max(df.shape[1] - 1, 0)

    fe = FeatureExtract(dir_path)
    structures, extra_features, labels = get_structures_extra_labels(df, path, n_extra_features)
    df_features = fe.get_features(
        structures,
        labels,
        data_extra=extra_features,
        extra_columns=list(df.columns)[1:-1] if n_extra_features > 0 else None,
        save=False,
        select_col=pd.read_csv(predict_path).columns.iloc[:-1],
        picture_feature=False,
        with_label=True,
    )


def predict_band_gap_hse(structures, extra_features, extra_cols, model: str = 'GAP_HSE_PBE') -> list[float]:
    """
    :param: structures
    :param: model
    :return [gap, gap, ...], gap == -1 represents the prediction is failed
    """
    import os

    import torch
    import joblib
    import numpy as np
    import pandas as pd
    import torch.nn.functional as F

    from utils.feature_utils import FeatureExtract
    from models.tdc_net import simple_net
    from models.classifier import get_model
    from utils.training_utils import load_checkpoint
    from config import MODEL_DICT, DEVICE

    if model not in MODEL_DICT:
        raise ValueError('Select a model to work.')
    data_file = f'{MODEL_DICT[model]}/train_data.zip'
    ckpt_file = f'{MODEL_DICT[model]}/best_model.ckpt'
    scaler_file = f'{MODEL_DICT[model]}/scaler.joblib'
    if not os.path.exists(data_file):
        raise FileNotFoundError(data_file)
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(ckpt_file)
    if not os.path.exists(scaler_file):
        raise FileNotFoundError(scaler_file)

    # Define the classifier
    classifier_model = get_model(load_file='models')

    # Define the regressor
    df = pd.read_csv(data_file)
    columns = list(df.columns)
    columns.pop()
    regressor_model = simple_net(
        num_features=len(columns),
        in_channels=10,
        prior_func=lambda x: F.relu6(x) * 8 / 6,
    )
    regressor_model.to(DEVICE)
    regressor_model.train()
    load_checkpoint(ckpt_file, regressor_model, inplace=True)

    # Extract the features
    fe = FeatureExtract()
    features = fe.get_features(
        structures,
        data_extra=extra_features,
        extra_columns=extra_cols,
        select_col=columns,
        with_label=False,
    )
    X_structures = np.stack([item for item in features['STRUCTURE']], axis=0)
    X_tabular = features.iloc[:, :-1].to_numpy()
    scaler = joblib.load(scaler_file)

    # Predict!!!
    y_pred = []
    y_is_metal = classifier_model.predict(X_tabular)
    X_structures, X_tabular = (torch.from_numpy(X_structures).to(device=DEVICE, dtype=torch.float32),
                               torch.from_numpy(scaler.transform(X_tabular)).to(device=DEVICE, dtype=torch.float32))
    y_gap_pred = regressor_model.forward(X_structures, X_tabular)
    for idx, (is_metal, gap) in enumerate(zip(y_is_metal, y_gap_pred)):
        if is_metal and gap <= 1:
            y_pred.append(.0)
        elif not is_metal and gap > 0:
            y_pred.append(float(gap))
        else:
            # Tow model give contradictory results
            print(
                f'Failed result: {float(gap)}eV, but {'is metal' if is_metal else 'is nonmetal'}, PBE gap{extra_features[idx][0]}'
            )
            y_pred.append(-1.0)

    return y_pred


if __name__ == '__main__':
    from utils.data_utils import get_data_from_db
    from utils.plot_utils import plot_corr_scatter

    data, _ = get_data_from_db(
        'datasets/c2db.db',
        select={
            'selection': 'gap',
            'filter': lambda row: getattr(row, 'magmom', 0) == 0
                                  and getattr(row, 'hform', 1) <= 0
                                  and getattr(row, 'ehull', 1) <= 0.05
                                  and getattr(row, 'dynamic_stability_stiffness', False)
                                  and getattr(row, 'dynamic_stability_phonons', False)
                                  and getattr(row, 'natoms') < 6
                                  and getattr(row, 'gap_hse', None) is None
            # and abs(row.gap_dir - row.gap) <= 1e-5
        },
        target='gap_nosoc',
        max_size=96 ** 2
    )
    structures = [item[0] for item in data]
    extra_features = [[item[1]] for item in data]
    extra_cols = ['gap_nosoc']

    num = 0
    dc = {}
    for idx, gap in enumerate(predict_band_gap_hse(structures, extra_features, extra_cols)):
        structure = structures[idx]
        dc[f'{structure.composition.reduced_formula}_{structure.get_space_group_info()[1]}'] = (
            gap, extra_features[idx][0])
        if 0.6 <= gap <= 2.0:
            num += 1
            print(structure.formula, gap)
    plot_corr_scatter([i[1] for i in dc.values()], [i[0] for i in dc.values()], 1600)
    print(num)
