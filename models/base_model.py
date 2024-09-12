# !/usr/bin/python
# coding:utf-8
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init


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
        self.hidden1 = nn.Linear(512, 2048)
        self.output_fc = nn.Linear(2048, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = F.relu(self.input_fc(x))
        out = self.dropout(out)
        out = F.relu(self.hidden1(out))
        out = self.dropout(out)
        out = self.output_fc(out)
        return out
