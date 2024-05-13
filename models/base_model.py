# !/usr/bin/python
# coding:utf-8
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from models.resnet import Bottleneck3D


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            # Kaiming/He init
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Xavier/Glorot
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    for m in model.modules():
        if isinstance(m, Bottleneck3D) and m.bn3.weight is not None:
            nn.init.constant_(m.bn3.weight, 0)


class MLP(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.input_fc = nn.Linear(num_features, 32)
        self.hidden = nn.Linear(32, 128)
        self.output_fc = nn.Linear(128, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.mish(self.input_fc(x))
        out = self.dropout(out)
        out = F.mish(self.hidden(out))
        out = self.dropout(out)
        out = self.output_fc(out)
        return F.relu(out)


class HierarchicalModel(nn.Module):
    def __init__(self, num_features, range_of_class=((0, 2), (2, 4), (4, 6), (6, 10))):
        super().__init__()
        self.range_of_class = range_of_class
        self.input_fc = nn.Linear(num_features, 32)
        self.hidden = nn.Linear(32, 128)
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(128, len(range_of_class))
        self.regressors = nn.ModuleList([nn.Linear(128, 1) for _ in range(len(range_of_class))])

    def forward(self, x):
        x = F.mish(self.input_fc(x), inplace=True)
        x = self.dropout(x)
        x = F.mish(self.hidden(x), inplace=True)
        x = self.dropout(x)

        classification_logits = self.classifier(x)
        classification_probs = F.softmax(classification_logits, dim=1)

        _, max_indices = classification_probs.max(dim=1)

        regression_outputs = torch.zeros(x.size(0), 1, device=x.device)
        for i, regressor in enumerate(self.regressors):
            mask = max_indices == i
            if mask.any():
                selected_x = x[mask]
                output = regressor(selected_x)
                output = torch.clamp(output, min=self.range_of_class[i][0], max=self.range_of_class[i][1])
                regression_outputs[mask] = output

        return classification_logits, regression_outputs.squeeze(1)


class ApplyTransformToEachLayer(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, img):
        # img(C, H, W)
        transformed_layers = []
        seed = torch.initial_seed()

        for i in range(img.shape[0]):
            random.seed(seed)
            torch.manual_seed(seed)

            layer = self.transform(img[i, ...].unsqueeze(0))
            transformed_layers.append(layer.squeeze(0))

        img_transformed = torch.stack(transformed_layers)
        return img_transformed


if __name__ == '__main__':
    from utils.data_utils import MlpDataset, get_data_from_db
    from utils.training_utils import train_and_eval

    data = get_data_from_db(
        '../datasets/c2db.db',
        select={'has_asr_hse': True},
        target=['results-asr.hse.json', 'kwargs', 'data', 'gap_hse_nosoc']
    )
    dataset = MlpDataset(data)

    train, val = torch.utils.data.random_split(dataset, (0.8, 0.2))

    model = HierarchicalModel(8)
    initialize_weights(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = [torch.nn.CrossEntropyLoss(), torch.nn.HuberLoss()]

    train = torch.utils.data.DataLoader(train, batch_size=32)
    val = torch.utils.data.DataLoader(val, batch_size=32)
    train_and_eval(model, train, val, criterion, optimizer, num_epochs=1000)
