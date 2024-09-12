import os.path
import warnings

import shap
import numpy as np
import pandas as pd
import joblib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import roc_curve, roc_auc_score, make_scorer, accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

SEED = 6
GAP_THRESHOLD = 1
WEIGHT = (36, 9)


def train(X_train, y_train, y_label, model) -> Pipeline:
    sample_weight = np.where(
        y_label == 0,
        WEIGHT[1],
        np.where(y_label >= GAP_THRESHOLD, WEIGHT[0], 1)
    )

    model.fit(X_train, y_train, xgbclassifier__sample_weight=sample_weight)

    joblib.dump(model, 'model.pkl')

    return model


def get_model(model_params: dict = None, load_file: str = None, seed=SEED) -> Pipeline:
    if model_params is None:
        new_model_params = {}
    else:
        new_model_params = model_params.copy()
        params = list(model_params.items())
        for key, value in params:
            if key.startswith('xgbclassifier__'):
                new_model_params[key[15:]] = value
                del new_model_params[key]

    model = make_pipeline(
        MinMaxScaler(),
        XGBClassifier(random_state=seed, **new_model_params)
    )

    if load_file and os.path.exists(f'{load_file}/model.pkl'):
        print('Load xgbclassifier model successfully!')
        model = joblib.load(f'{load_file}/model.pkl')

    return model


def grid_search_cv(X_train, y_train, y_label, model, cv=5) -> dict:
    params = {
        'xgbclassifier__learning_rate': [0.01, 0.1],
        'xgbclassifier__n_estimators': [50, 200],
        'xgbclassifier__max_depth': [3, 6],
        'xgbclassifier__min_child_weight': [1, 3, 6],
        'xgbclassifier__gamma': [0, 0.1, 0.2],
        'xgbclassifier__subsample': [0.8, 1.0],
        'xgbclassifier__colsample_bytree': [0.8, 1.0],
        'xgbclassifier__reg_lambda': [1e-5, 1e-4],
        'xgbclassifier__reg_alpha': [0, 0.01],
        # 'xgbclassifier__scale_pos_weight': [1, 5, 10],  # Useful for imbalanced data
        # 'xgbclassifier__booster': ['gbtree', 'gblinear', 'dart'],
        # 'xgbclassifier__tree_method': ['auto', 'exact', 'approx', 'hist'],  # For controlling tree-building strategy
    }

    grid_search = GridSearchCV(model, param_grid=params, cv=cv, n_jobs=-1, verbose=3, scoring='accuracy')
    sample_weight = np.where(
        y_label == 0,
        WEIGHT[1],
        np.where(y_label >= GAP_THRESHOLD, WEIGHT[0], 1)
    )
    grid_search.fit(X_train, y_train, xgbclassifier__sample_weight=sample_weight)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    return grid_search.best_params_


def evaluate_cv(X, y, model, *, cv=5, seed=SEED):
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'auc': make_scorer(roc_auc_score, needs_proba=True)
    }

    results = cross_validate(model, X, y, cv=stratified_kfold, scoring=scoring, return_train_score=False)

    avg_accuracy = results['test_accuracy'].mean()
    avg_auc = results['test_auc'].mean()

    return avg_accuracy, avg_auc


def test(X_test, y_test, model) -> tuple:
    score = model.score(X_test, y_test)

    y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = round(roc_auc_score(y_test, y_pred), 4)

    return score, auc


def evaluate(X, model, y=None, y_label=None):
    # if y or y_label is None, only return results
    # return tuple(results, wrong predict, wrong label)
    results = model.predict(X)

    if y_label is None or y is None:
        return results

    # Find indices of wrong predictions
    wrong_indices = (results != y).nonzero()[0]

    # Get wrong predictions and corresponding true labels
    wrong_preds = results[wrong_indices]
    wrong_labels = y_label[wrong_indices]

    return results, wrong_preds, wrong_labels


def plot_roc_auc(X_test, y_test, model):
    y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    auc = round(roc_auc_score(y_test, y_pred), 4)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()


def plot_shap(X_train, model, feature_names):
    X = model.named_steps['minmaxscaler'].transform(X_train)

    explainer = shap.TreeExplainer(model.named_steps['xgbclassifier'])
    shap_values = explainer.shap_values(X)

    shap.summary_plot(shap_values, X, feature_names=feature_names)

    mean_abs_shap_values = np.abs(shap_values).mean(axis=0)

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap_values
    })

    top_10_features = feature_importance_df.sort_values(by='Mean_Abs_SHAP', ascending=False).head(10)
    print("Top 10 important features:")
    print(top_10_features)


def split_datasets(X, y, stratify=None, test_size=0.2, seed=SEED):
    if stratify is None:
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, stratify=stratify)

    return X_train, X_test, y_train, y_test


def main():
    df = pd.read_csv('../datasets/hse_set_pbe/train_data.zip')
    df['gap_nosoc'] = np.where(df['gap_nosoc'] == 0, 0, np.where(df['gap_nosoc'] <= 1, 1, 2))
    df = df[[
        'gap_nosoc',
        'P_E_MEAN',
        'GAMMA',
        'MAX_OXIDATION_STATE_MEAN',
        'VALENCE_STD',
        'B',
        'A',
        'IONIZATION_ENERGIES_STD',
        'ELECTRON_AFFINITIES_MIN',
        'IONIC_RADII_MIN',
        'LABEL'
    ]]

    X, y = df.iloc[:, :-1].to_numpy(), (df['LABEL'] <= 1e-5).to_numpy()
    _, _, y_train_label, y_test_label = split_datasets(X, df['LABEL'].to_numpy(), stratify=y, seed=SEED)

    X_train, X_test, y_train, y_test = split_datasets(X, y, stratify=y, seed=SEED)

    model = get_model(load_file='./')
    # model_params = grid_search_cv(X, y, df['LABEL'].to_numpy(), model)
    # model = get_model(model_params)
    train(X_train, y_train, y_train_label, model)

    score, _ = test(X_test, y_test, model)
    print(f'score: {score}')

    print(*evaluate_cv(X, y, model))

    # plot_roc_auc(X_test, y_test, model)
    # plot_shap(X_train, model, feature_names=list(df.columns)[:-1])


if __name__ == '__main__':
    main()
