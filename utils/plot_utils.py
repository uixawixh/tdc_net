from typing import Union, Tuple

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.eval_utils import predict_value

custom_style = {
    'font.size': 16,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',
}

plt.style.use(custom_style)


def get_features_and_targets(dataset, columns):
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    assert len(dataset) > 0
    features, targets = next(iter(dataloader))

    if isinstance(features, torch.Tensor):
        features = features.cpu()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu()
    df = pd.DataFrame(features.numpy(), columns=columns)
    df['target'] = targets.numpy()
    return df


def feature_box(dataset, columns=None):
    df = get_features_and_targets(dataset, columns)
    df_numeric = df.select_dtypes(include=[float, int])

    plt.figure(figsize=(12, 6), dpi=240)
    plt.boxplot(df_numeric.values, labels=df_numeric.columns, patch_artist=True)  # 使用patch_artist填充箱体颜色
    plt.xticks(rotation=45)
    plt.title('Feature Distribution Boxplots')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5), dpi=220)

    epochs = list(range(1, len(train_losses) + 1))
    data = {
        'Epochs': epochs * 2,
        'Loss': train_losses + val_losses,
        'Type': ['Train Loss'] * len(train_losses) + ['Validation Loss'] * len(val_losses)
    }

    sns.lineplot(x='Epochs', y='Loss', hue='Type', data=data, palette='Blues')

    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close()


def plot_violin(xx, title, xlabel):
    plt.figure(figsize=(10, 6), dpi=220)
    sns.violinplot(xx, cut=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()
    plt.close()


def plot_hist(xx, title, xlabel):
    plt.figure(figsize=(10, 6), dpi=220)
    sns.histplot(xx, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.show()
    plt.close()


def image_plot(image):
    _, D, _, _ = image.shape

    def normalize_channel(img):
        img_min, img_max = img.min(), img.max()
        return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    fig, axs = plt.subplots(1, D, figsize=(D * 5, 5), dpi=240)

    if D == 1:
        axs = np.expand_dims(axs, 0)

    for depth in range(D):
        assert image.shape[0] == 3, "There must be 3 channels for RGB color."

        normalized_image = np.stack([normalize_channel(image[c, depth, ...]) for c in range(3)], axis=-1)
        axs[depth].imshow(normalized_image)
        axs[depth].axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_atoms(return_matrix):
    if isinstance(return_matrix, torch.Tensor):
        return_matrix.requires_grad = False
        return_matrix = return_matrix.detach().cpu().numpy()
    n = return_matrix.shape[1]

    # Get the figure and axes objects
    # Layout is 3 rows (one for each feature) and n columns (one for each layer)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 15), dpi=200)

    # Titles for each channel (feature map)
    titles = ['Electronegativity', 'Period', 'Group']

    # Use nested loops to plot the matrices
    for feature_idx in range(3):  # Loop through the number of features
        for layer_idx in range(n):  # Loop through layers
            ax = axes[feature_idx, layer_idx] if n > 1 else axes[feature_idx]
            feature_map = return_matrix[feature_idx, layer_idx]
            max_val = feature_map.max()  # Find the maximum value for normalization
            normalized_matrix = feature_map / max_val if max_val > 0 else feature_map

            # Display the image
            ax.imshow(normalized_matrix, cmap='viridis')

            # Only set titles for the top row
            if feature_idx == 0:
                ax.set_title(f'Layer {layer_idx + 1}')

            ax.axis('off')  # Turn off axis labels

    # Setting feature title for each row
    for feature_idx, title in enumerate(titles):
        if n == 1:
            axes[feature_idx].set_ylabel(title, fontsize=12)
        else:
            axes[feature_idx, 0].set_ylabel(title, fontsize=12)

    plt.tight_layout()
    plt.show()
    plt.close()


def plot_true_predict_model(model, dataloader: Union[DataLoader, Tuple], model_load_path: str = None):
    # If dataloader is tuple, then plot train and test prediction in a figure
    if model_load_path is not None:
        model.load_state_dict(torch.load(model_load_path)['model_state_dict'])

    if not isinstance(dataloader, (tuple, list)):
        dataloader = (dataloader,)

    for item, label in zip(dataloader, ('train', 'test')):
        if isinstance(item, str):
            item = torch.load(item)

        true_values, pred_values = predict_value(model, item)

        plt.figure(figsize=(8, 8), dpi=220)
        sns.scatterplot(x=true_values, y=pred_values, label=label, color='#ef8a43' if label == 'test' else 'blue')
        plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.show()
    plt.close()


def feature_corr(dataset, columns=None):
    """dataset must be single column feature."""
    df = get_features_and_targets(dataset, columns)
    correlation_matrix = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10), dpi=240)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(correlation_matrix, mask=mask, cmap=plt.cm.Blues, annot=True,
                square=True, cbar_kws={"shrink": .5}, ax=ax)

    # Rotate the x-axis labels to 45 degrees and y-axis labels to horizontal
    ax.set_xticklabels(ax.get_xmajorticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_ymajorticklabels(), rotation=0)

    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    plt.show()
    plt.close()

    # Print sorted target correlations
    target_correlation = correlation_matrix["target"].drop("target").sort_values(ascending=False)
    print(target_correlation)


if __name__ == '__main__':
    from data_utils import MlpDataset, get_data_from_db

    data = get_data_from_db(
        '../datasets/c2db.db',
        # select={'has_asr_hse': True},
        # target=['results-asr.hse.json', 'kwargs', 'data', 'gap_hse_nosoc']
        select={},
        target=['results-asr.gs.json', 'kwargs', 'data', 'gap_nosoc']
    )
    plot_hist(list(map(lambda x: x[1], data)), '', 'gap[eV]')
    # dataset = MlpDataset(data)
    # feature_corr(
    #     dataset,
    #     columns=[
    #         'density',
    #         'electroneg',
    #         'electronaff',
    #         'ionenergy',
    #         'radius',
    #         'electrons',
    #         'atoms',
    #         'spacegroup'
    #     ]
    # )
