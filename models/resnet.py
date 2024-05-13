# !/usr/bin/python
# coding:utf-8
from typing import Callable

import torch.nn.functional as F
import torch.optim
from torch import nn


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck3D, self).__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride),
                               padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes * self.expansion, kernel_size=1, stride=(1, stride, stride), bias=False),
                nn.BatchNorm3d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet3D(nn.Module):
    def __init__(
            self,
            block,
            num_blocks,
            num_features,
            in_channels=3,
            prior_func=None
    ):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, self.in_planes,
                               kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)

        # Creating layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Regression head
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()
        self.fusion_fc = nn.Linear(128 + 512 * block.expansion, 1)

        self.feat_fc = nn.Linear(num_features, 32)
        self.hidden_fc = nn.Linear(32, 128)
        self.prior_func = prior_func

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]

        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1))

        return nn.Sequential(*layers)

    def forward(self, x, features):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.flatten(out)

        # Process additional features
        feat_out = F.dropout(F.mish(self.feat_fc(features)))
        feat_out = F.dropout(F.mish(self.hidden_fc(feat_out)))
        combined_features = torch.cat((out, feat_out), dim=1)
        out = self.fusion_fc(combined_features)

        if self.prior_func is not None:
            out = self.prior_func(out)
        return out


def resnet50(num_features: int, in_channels: int = 3, prior_func: Callable = None):
    return ResNet3D(Bottleneck3D, [3, 4, 6, 3], num_features, in_channels, prior_func)


def resnet101(num_features: int, in_channels: int = 3, prior_func: Callable = None):
    return ResNet3D(Bottleneck3D, [3, 4, 23, 3], num_features, in_channels, prior_func)


if __name__ == '__main__':
    from base_model import initialize_weights
    from utils.training_utils import train_and_eval
    from utils.data_utils import get_dataloader_v1

    # Prior >=0, but need to avoid nn dead
    model = resnet50(num_features=8, prior_func=lambda x: F.leaky_relu(x, 0.01))
    initialize_weights(model)
    train_loader, val_loader, test_loader = get_dataloader_v1(
        '../datasets/c2db.db',
        save_path='../datasets/tdcnet_v1',
        batch_size=16,
        select={'has_asr_hse': True},
        target=['results-asr.hse.json', 'kwargs', 'data', 'gap_hse_nosoc']
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.HuberLoss()

    train_and_eval(model, train_loader, val_loader, criterion, optimizer,
                   checkpoint_path='../checkpoints', start_epoch=1, num_epochs=60)
