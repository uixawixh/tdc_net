# !/usr/bin/python
# coding:utf-8
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init

from models.tdc_net import Bottleneck3D


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


class MLP(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.input_fc = nn.Linear(num_features, 512)
        self.hidden = nn.Linear(512, 2048)
        self.output_fc = nn.Linear(2048, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.input_fc(x))
        out = self.dropout(out)
        out = F.relu(self.hidden(out))
        out = self.dropout(out)
        out = self.output_fc(out)
        return F.relu6(out) * 8 / 6


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


if __name__ == '__main__':
    from utils.data_utils import MlpDataset, get_data_from_db
    from utils.training_utils import train_and_eval
    from utils.plot_utils import plot_true_predict_model

    data = get_data_from_db(
        '../datasets/c2db.db',
        select={'has_asr_hse': True},
        target=['results-asr.hse.json', 'kwargs', 'data', 'gap']
    )
    dataset = MlpDataset(data)
    torch.manual_seed(1007)
    train, val = torch.utils.data.random_split(dataset, (0.8, 0.2))

    model = MLP(76)
    initialize_weights(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.01)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20, 4, 0)

    train = torch.utils.data.DataLoader(train, batch_size=64)
    val = torch.utils.data.DataLoader(val, batch_size=64)
    train_and_eval(model, train, val, criterion, optimizer, scheduler=scheduler, num_epochs=300)
    plot_true_predict_model(model, (train, val))
