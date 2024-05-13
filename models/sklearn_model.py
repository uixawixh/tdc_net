import os
import json

import h2o
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from h2o.automl import H2OAutoML
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV

from config import SEED, CPU_CORE
from utils.sklearn_utils import get_all_dataset

custom_style = {
    'font.size': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',
}

plt.style.use(custom_style)


def smoter(X, y, target_feature_range, n_samples=100, n_neighbors=8):
    np.random.seed(SEED)
    X, y = np.asarray(X), np.asarray(y)
    indices = np.where((y >= target_feature_range[0]) & (y <= target_feature_range[1]))[0]
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
    nn.fit(X[indices])
    X_new = []
    y_new = []

    while len(X_new) < n_samples:
        idx = np.random.choice(indices)
        _, neighbors_idx = nn.kneighbors(X[idx].reshape(1, -1), return_distance=True)
        neighbor_idx = np.random.choice(neighbors_idx[0][1:])
        diff = X[indices][neighbor_idx] - X[idx]
        gap = np.random.rand(1)

        synth_sample_X = X[idx] + gap * diff
        synth_sample_y = y[idx] + gap * (y[indices][neighbor_idx] - y[idx])

        X_new.append(synth_sample_X)
        y_new.append(synth_sample_y)

    return np.vstack([X, np.array(X_new)]), np.concatenate([y, np.array(y_new).reshape(-1)])


class SklearnPredictor:

    def __init__(self, dataset):
        self.dataset = dataset
        self.models = {
            "Polynomial Ridge Regression": make_pipeline(
                PolynomialFeatures(),
                Ridge()
            ),
            "K Neighbors Regressor": KNeighborsRegressor(n_jobs=CPU_CORE),
            "Support Vector Machine": SVR(),
            "Random Forest": RandomForestRegressor(n_jobs=CPU_CORE),
            "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
        }
        self.auto = None
        self.scores = {}
        self.train_preds = {}
        self.val_preds = {}
        self.X, self.y = None, None

        # init hyperparameters
        hyperparams_path = 'sklearn_hyperparams.json'
        if os.path.isfile(hyperparams_path):
            with open(hyperparams_path, 'r') as fp:
                hyperparams = json.load(fp)

            for name, params in hyperparams.items():
                if name not in self.models:
                    continue

                if isinstance(self.models[name], Pipeline):
                    steps = self.models[name].steps
                    for step_name, step_params in params.items():
                        for step in steps:
                            step[0] == step_name and step[1].set_params(**step_params)
                else:
                    self.models[name].set_params(**params)

    def cross_validate(self):
        if self.X is None and self.y is None:
            self.X, self.y = get_all_dataset(self.dataset)
        X, y = self.X, self.y
        scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        for name, model in self.models.items():
            scores = cross_validate(model, X, y, cv=5, scoring=scoring_metrics, n_jobs=CPU_CORE)

            # Negative MSE and MAE are returned by cross_validate, so convert to positive
            mse_scores = -scores['test_neg_mean_squared_error']
            mae_scores = -scores['test_neg_mean_absolute_error']
            r2_scores = scores['test_r2']
            mean_mse, mean_mae, mean_r2 = mse_scores.mean(), mae_scores.mean(), r2_scores.mean()
            std_mse, std_mae, std_r2 = mse_scores.std(), mae_scores.std(), r2_scores.std()
            self.scores[name] = {
                "MSE": mse_scores,
                "MAE": mae_scores,
                "R2": r2_scores,
            }

            print(f"{name} - MSE: {mean_mse:.2f}, ±{std_mse:.2f}")
            print(f"{name} - MAE: {mean_mae:.2f}, ±{std_mae:.2f}")
            print(f"{name} - R2: {mean_r2:.2f}, ±{std_r2:.2f}")

        metrics_data = []
        for model_name, metrics_dict in self.scores.items():
            for metric_name, values in metrics_dict.items():
                value = values.mean()
                error = values.std()

                metrics_data.append({
                    'Model': ''.join(map(lambda x: x if x.isupper() else '', model_name)),
                    'Metric': metric_name,
                    'Value': value,
                    'Error': error,
                })

        plt.figure(figsize=(12, 7), dpi=220)
        metrics_df = pd.DataFrame(metrics_data)
        sns_barplot = sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_df, capsize=.1, palette='Blues')
        bars = sns_barplot.patches
        errors = metrics_df['Error'].values
        for bar, error in zip(bars, errors):
            x_center = bar.get_x() + bar.get_width() / 2
            y_value = bar.get_height()
            plt.errorbar(x_center, y_value, yerr=error, fmt='none', c='black', capthick=2, capsize=5)

        plt.title('MSE, MAE and R2 of Different Models')
        plt.ylabel('Score')
        plt.gca().set_xlabel('')
        plt.tight_layout()
        plt.show()
        plt.close()

    def train_and_evaluate(self, use_smoter: bool = False):
        if self.X is None and self.y is None:
            self.X, self.y = get_all_dataset(self.dataset)

        models = self.models.copy()
        X, y = self.X, self.y
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
        if use_smoter:
            X_train, y_train = smoter(X_train, y_train, target_feature_range=(6, 10))
        if self.auto is not None:
            models['AUTOML'] = self.auto

        for name, model in models.items():
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            self.train_preds[name] = train_pred
            self.val_preds[name] = val_pred

            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)

            print(f"{name} - Train MSE: {train_mse:.4f}, Validation MSE: {val_mse:.4f}")
            print(f"{name} - Train MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}")
            print(f"{name} - Train R2: {train_r2:.4f}, Validation R2: {val_r2:.4f}")

        self.visualize_predictions(y_train, y_val)

    def visualize_predictions(self, y_train, y_val):
        y_train, y_val = np.asarray(y_train), np.asarray(y_val)
        _, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=220)

        for i, (name, model) in enumerate(self.models.items()):
            if y_train.ndim > 1:
                y_train = y_train.ravel()
            train_preds = self.train_preds[name].ravel() if self.train_preds[name].ndim > 1 else self.train_preds[name]
            if y_val.ndim > 1:
                y_val = y_val.ravel()
            val_preds = self.val_preds[name].ravel() if self.val_preds[name].ndim > 1 else self.val_preds[name]

            row, col = divmod(i, 3)
            sns.scatterplot(x=y_train, y=train_preds, ax=axs[row, col], label='train')
            axs[row, col].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], '--r')
            axs[row, col].set_title(f'{name}')
            axs[row, col].set_xlabel('True Values')
            axs[row, col].set_ylabel('Predictions')
            axs[row, col].set_ylim(0, 10)
            axs[row, col].set_xlim(0, 10)

            sns.scatterplot(x=y_val, y=val_preds, ax=axs[row, col], color='#ef8a43', label='validation')

        plt.tight_layout()
        plt.legend()
        plt.show()
        plt.close()

    def grid_search_cv(self, cv: int = 5):
        if self.X is None and self.y is None:
            self.X, self.y = get_all_dataset(self.dataset)
        param_grids = {
            "Polynomial Ridge Regression": {
                'polynomialfeatures__degree': [2, 3],
                'ridge__alpha': [0.1, 1, 10]
            },
            "K Neighbors Regressor": {
                'n_neighbors': [3, 5, 7],
                'leaf_size': [20, 30, 40]
            },
            "Support Vector Machine": {
                'degree': [2, 3],
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            },
            "Random Forest": {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
            },
            "Gradient Boosting": {
                'n_estimators': [10, 50, 100, 200],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2'],
                'subsample': [0.9, 1.0, 1.1]
            }
        }

        best_params = {}
        for name, model in self.models.items():
            grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='neg_mean_squared_error',
                                       n_jobs=CPU_CORE)
            grid_search.fit(self.X, self.y)
            best_params[name] = grid_search.best_params_
            print(f"Best parameters for {name}: {grid_search.best_params_}, MSE: {-grid_search.best_score_:.3f}")

        # 将最佳参数写入文件
        with open('sklearn_hyperparams.json', 'w') as f:
            json.dump(best_params, f, indent=4)

    def automl(self):
        h2o.init()
        if self.X is None and self.y is None:
            self.X, self.y = get_all_dataset(self.dataset)

        h2oX = h2o.H2OFrame(self.X)
        h2oy = h2o.H2OFrame(self.y)
        if len(h2oy.columns) > 1:
            h2oy = h2oy.as_data_frame().values.flatten()
            h2oy = h2o.H2OFrame(h2oy)
        h2o_train_frame = h2oX.cbind(h2oy)
        h2o_train_frame.set_names(list(f'C{i}' for i in range(self.X.shape[1])) + ['target'])

        model = H2OAutoML(max_models=30, max_runtime_secs=3600, seed=SEED, nfolds=5)
        model.train(x=list(f'C{i}' for i in range(self.X.shape[1])), y='target', training_frame=h2o_train_frame)
        lb = model.leaderboard
        lb_df = lb.as_data_frame()
        lb_df.to_csv("automl_leaderboard.csv", index=True)
        self.auto = model.leader
        h2o.save_model(model=self.auto, path="automl_best_model", force=True)

    def tsne(self, perplexity=30, n_iter=1000):
        if self.X is None and self.y is None:
            self.X, self.y = get_all_dataset(self.dataset)
        X_tsne = TSNE(
            n_components=2,
            random_state=SEED,
            perplexity=perplexity,
            n_iter=n_iter,
            init='pca',
            learning_rate='auto',
            n_jobs=CPU_CORE,
        ).fit_transform(self.X)

        gap_categories = np.digitize(self.y, bins=[3, 6])
        gap_labels = np.array(['0-3eV', '3-6eV', '>6eV'])
        label_mapping = dict(zip(range(len(gap_labels)), gap_labels))
        gap_category_labels = np.vectorize(label_mapping.get)(gap_categories)

        plt.figure(figsize=(10, 10), dpi=220)
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=gap_category_labels, palette='Set1',
                        hue_order=gap_labels)
        plt.title("t-SNE Visualization of Band Gaps")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")

        plt.show()
        plt.close()

    def predict(self, X_new):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_new)
        return predictions


if __name__ == '__main__':
    from utils.data_utils import MlpDataset, get_data_from_db

    data = get_data_from_db(
        '../datasets/c2db.db',
        select={'has_asr_hse': True},
        target=['results-asr.hse.json', 'kwargs', 'data', 'gap_hse_nosoc']
    )
    dataset = MlpDataset(data)
    sp = SklearnPredictor(dataset)
    sp.automl()
