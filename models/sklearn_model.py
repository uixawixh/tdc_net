import os
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler, minmax_scale, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from xgboost import XGBRegressor

from config import SEED
from utils.feature_utils import FeatureExtract
from utils.sklearn_utils import get_all_dataset

custom_style = {
    'font.size': 20,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',
}

plt.style.use(custom_style)


class SklearnPredictor:

    def __init__(self, dataset, direct_data=False):
        self.dataset = dataset
        self.models = {
            "Ridge Regression": make_pipeline(
                StandardScaler(),
                PolynomialFeatures(),
                Ridge()
            ),
            "KNN": make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(n_jobs=-1)
            ),
            "SVR": make_pipeline(
                StandardScaler(),
                SVR()
            ),
            "Random Forest": make_pipeline(
                StandardScaler(),
                RandomForestRegressor(n_jobs=-1)
            ),
            "GBDT": make_pipeline(
                StandardScaler(),
                GradientBoostingRegressor(random_state=SEED)
            ),
            "XGBoost": make_pipeline(
                StandardScaler(),
                XGBRegressor(random_state=SEED, n_jobs=-1)
            )
        }
        self.auto = None
        self.scores = {}
        self.train_preds = {}
        self.val_preds = {}
        self.X, self.y = None, None
        if direct_data:
            self.X, self.y = dataset[:, :-1], dataset[:, -1]

        self.init_model()

    def init_model(self):
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
            scores = cross_validate(model, X, y, cv=5, scoring=scoring_metrics, n_jobs=-1)

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

    def train_and_evaluate(self):
        if self.X is None and self.y is None:
            self.X, self.y = get_all_dataset(self.dataset)

        models = self.models.copy()
        X, y = self.X, self.y
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=SEED)
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
            axs[row, col].plot([y_train.min() - 10, y_train.max() + 10], [y_train.min() - 10, y_train.max() + 10],
                               '--r')
            axs[row, col].set_title(f'{name}')
            axs[row, col].set_xlabel('True Values')
            axs[row, col].set_ylabel('Predictions')
            axs[row, col].set_ylim(0, 8)
            axs[row, col].set_xlim(0, 8)
            axs[row, col].grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')

            sns.scatterplot(x=y_val, y=val_preds, ax=axs[row, col], color='#ef8a43', label='validation')

        plt.tight_layout()
        axs[0, 0].legend()
        plt.show()
        plt.close()

    def grid_search_cv(self, cv: int = 5):
        if self.X is None and self.y is None:
            self.X, self.y = get_all_dataset(self.dataset)
        param_grids = {
            "Ridge Regression": {
                'ridge__alpha': [0.1, 0.5, 1, 5, 10, 30],
                'polynomialfeatures__degree': [1, 2],
            },
            "KNN": {
                'kneighborsregressor__n_neighbors': [3, 5, 7],
                'kneighborsregressor__leaf_size': [20, 30, 40],
                'kneighborsregressor__weights': ['uniform', 'distance'],
                'kneighborsregressor__metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
            },
            "SVR": {
                'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'svr__degree': [2, 3, 4],
                'svr__C': [0.1, 1, 10],
            },
            "Random Forest": {
                'randomforestregressor__n_estimators': [10, 100, 500],
                'randomforestregressor__max_depth': [None, 5, 10],
                'randomforestregressor__min_samples_split': [2, 4, 6],
                'randomforestregressor__min_samples_leaf': [1, 2, 4],
                'randomforestregressor__max_features': ['sqrt', 'log2'],
            },
            "GBDT": {
                'gradientboostingregressor__n_estimators': [10, 100, 500],
                'gradientboostingregressor__max_depth': [3, 5, 7],
                'gradientboostingregressor__min_samples_split': [2, 4, 6],
                'gradientboostingregressor__min_samples_leaf': [1, 2, 4],
                'gradientboostingregressor__max_features': ['sqrt', 'log2'],
                'gradientboostingregressor__subsample': [0.6, 0.8, 1.0]
            },
            "XGBoost": {
                'xgbregressor__learning_rate': [0.01, 0.1, 1.0],
                'xgbregressor__n_estimators': [50, 100, 500],
                'xgbregressor__max_depth': [3, 6, 10],
                'xgbregressor__min_child_weight': [1, 5],
                'xgbregressor__gamma': [0, 0.1, 0.2],
                'xgbregressor__subsample': [0.6, 0.8, 1.0],
                'xgbregressor__reg_lambda': [1e-5, 1e-4]
            }
        }

        best_params = {}
        for name, model in self.models.items():
            grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='r2', n_jobs=-1)
            grid_search.fit(self.X, self.y)
            best_params[name] = grid_search.best_params_
            print(f"Best parameters for {name}: {grid_search.best_params_}, R2: {grid_search.best_score_:.3f}")

        # 将最佳参数写入文件
        with open('sklearn_hyperparams.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        self.init_model()

    def feature_score(self, columns, n: int = 30, feature_select: bool = True):
        if self.X is None and self.y is None:
            self.X, self.y = get_all_dataset(self.dataset)
        estimator = RandomForestRegressor(n_estimators=1000, random_state=SEED, n_jobs=-1)
        estimator.fit(self.X, self.y)
        rf_importance = estimator.feature_importances_

        normalized_rf_importance = minmax_scale(rf_importance, feature_range=(0, 1))

        feature_scores_df = pd.DataFrame({
            'Feature': columns,
            'Total Score': normalized_rf_importance
        }).sort_values(by='Total Score', ascending=False)

        if feature_select:
            top_features = feature_scores_df.head(n)['Feature'].values
            top_features_indices = [columns.index(feature) for feature in top_features]
            print('top_features_indices:', top_features_indices)
            self.X = self.X[:, top_features_indices]

        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 8), dpi=220)

        ax = sns.barplot(x="Total Score", y="Feature", data=feature_scores_df[:n], color="skyblue")

        ax.set_title('Feature Importance Scores')
        ax.set_xlabel('Score')
        ax.set_ylabel('')
        plt.tight_layout()
        plt.show()
        plt.close()

        return feature_scores_df

    def automl(self):
        try:
            import h2o
            from h2o.automl import H2OAutoML
        except ImportError:
            print('No ML software h2o')
            return
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
            n_jobs=-1,
        ).fit_transform(self.X)

        plt.figure(figsize=(12, 10), dpi=220)
        ax = plt.gca()
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=self.y, palette='viridis', legend=False, ax=ax)
        plt.title("t-SNE Visualization of Band Gaps")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")

        norm = plt.Normalize(self.y.min(), self.y.max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax)

        plt.show()
        plt.close()

    def predict(self, X_new):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_new)
        return predictions


if __name__ == '__main__':
    from utils.data_utils import MlpDataset, get_data_from_db

    extra_features = [
        'efermi',
        'hform',
        'evac',
        'dos_at_ef_nosoc',
        'gap'
    ]
    data, extra = get_data_from_db(
        '../datasets/c2db.db',
        {'selection': 'gap_hse'},
        'gap_hse',
        *extra_features,
        max_size=96 ** 2
    )

    fx = FeatureExtract('.')
    df = fx.get_features([i[0] for i in data], [i[1] for i in data], data_extra=extra, extra_columns=extra_features,
                         picture_feature=False)
    y = df['LABEL']
    dataset = MlpDataset([[i, j] for i, j in zip(df.iloc[:, :-2].to_numpy(), y)])
    sp = SklearnPredictor(dataset)
    sp.feature_score(n=30, columns=list(fx.columns[:-2]))
    sp.grid_search_cv(5)
    # sp.cross_validate()
    sp.train_and_evaluate()
    # sp.tsne()
    # from utils.plot_utils import feature_corr, feature_box
    # from utils.data_utils import get_properties
    #
    # data = pd.read_csv('pce.csv')
    # properties_list = []
    #
    # for formula in data['formula']:
    #     properties = get_properties(formula)
    #     properties_list.append(properties)
    #
    # properties_df = pd.DataFrame(properties_list)
    # data = pd.concat([data, properties_df], axis=1)
    #
    # # features_df = data.apply(lambda row: get_features_from_components(row['elements']), axis=1)
    # # data = pd.concat([data, features_df], axis=1)
    # # data = data.groupby('elements', as_index=False).mean()
    # data.drop(['formula'], axis=1, inplace=True)
    #
    # y = data.iloc[:, 0:1].to_numpy(np.float32)
    # data.drop('PCE', axis=1, inplace=True)
    # # data.drop(['NIR wave laser(nm)', 'NIR intensity(W/cm2)', 'NIR time(min)'], inplace=True, axis=1)
    # # data.drop(['Decoration1', 'Decoration2', 'Decoration3'], inplace=True, axis=1)
    #
    # columns_to_encode = ['Decoration1', 'Decoration2', 'Decoration3']
    # for column in columns_to_encode:
    #     data[column] = pd.factorize(data[column])[0]
    # # data = pd.get_dummies(data, columns=['Decoration1', 'Decoration2', 'Decoration3'])
    # data.dropna(inplace=True)
    # X = data.to_numpy(np.float32)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # # from sklearn.decomposition import PCA
    # #
    # # pca = PCA(n_components=8)
    # # X = pca.fit_transform(X)
    #
    # from torch.utils.data import Dataset
    #
    #
    # class MlpDataset(Dataset):
    #
    #     def __init__(self, data):
    #         self.data = data
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    #     def __getitem__(self, item):
    #         features, target = self.data[item]
    #         return features, target
    # import torch
    # from models.base_model import MLP, initialize_weights
    # from utils.training_utils import train_and_eval
    #
    # dataset = MlpDataset([[i, j] for i, j in zip(X, y)])
    # torch.manual_seed(1007)
    # train, val = torch.utils.data.random_split(dataset, (0.8, 0.2))
    #
    # model = MLP(75)
    # initialize_weights(model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)
    # criterion = torch.nn.L1Loss()
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, 0)
    #
    # train = torch.utils.data.DataLoader(train, batch_size=1)
    # val = torch.utils.data.DataLoader(val, batch_size=8)
    # train_and_eval(model, train, val, criterion, optimizer, scheduler=scheduler, num_epochs=300)
    # feature_box(np.hstack([X, y]))
    # feature_corr(np.hstack([X, y]))

    # feature_box(np.hstack([X, y]))
    # sp = SklearnPredictor(np.hstack([X, y]), True)
    # # sp.grid_search_cv(5)
    # # sp.cross_validate()
    # sp.train_and_evaluate()
    # sp.tsne()
