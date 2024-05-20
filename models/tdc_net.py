# !/usr/bin/python
# coding:utf-8
from typing import Callable

import torch.nn.functional as F
import torch.optim
from torch import nn

from models.modules import AttentionLayer, SEBlock, SpatialAttention3D


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, channel_attention=False):
        super().__init__()

        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.se = SEBlock(planes * self.expansion)
        self.channel_attention = channel_attention

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.channel_attention:
            out = self.se(out)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet3D(nn.Module):
    def __init__(
            self,
            block,
            num_blocks,
            in_channels=3,
            channel_attention=False,
            spatial_attention=False,
    ):
        super().__init__()
        self.in_planes = 64
        self.channel_attention = channel_attention
        self.spatial_attention = spatial_attention

        # Initial convolution
        self.conv1 = nn.Conv3d(in_channels, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)

        # Creating layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Regression head
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = [block(self.in_planes, planes, stride)]

        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1))

        if self.spatial_attention:
            sa = SpatialAttention3D(self.in_planes, kernel_size=(5, 1, 1))  # depth attention
            layers.append(sa)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = self.flatten(out)

        return out


class VGG3D(nn.Module):
    def __init__(self, config, input_channels=3, output_dim=1):
        super().__init__()

        self.features = self._make_layers(config, input_channels)

        self.avgpool = nn.AdaptiveAvgPool3d((7, 7, 7))
        self.flatten = nn.Flatten()

        self.out_layer = nn.Sequential(
            nn.Linear(config[-1] * 7 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.out_layer(x)
        return x

    @staticmethod
    def _make_layers(config, input_channels):
        layers = []
        for v in config:
            if v == 'M':
                # Padding to avoid zero-size
                layers.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=(1, 0, 0)))
                continue
            out_channels = v
            layers.extend([
                nn.Conv3d(input_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ])
            input_channels = out_channels
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2, padding=(1, 0, 0)))
        return nn.Sequential(*layers)


class FireModule3D(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1x1_channels, expand3x3x3_channels):
        super().__init__()
        self.squeeze = nn.Conv3d(in_channels, squeeze_channels, kernel_size=1)
        self.expand1x1x1 = nn.Conv3d(squeeze_channels, expand1x1x1_channels, kernel_size=1)
        self.expand3x3x3 = nn.Conv3d(squeeze_channels, expand3x3x3_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.squeeze(x), inplace=True)
        return torch.cat([
            F.relu(self.expand1x1x1(x), inplace=True),
            F.relu(self.expand3x3x3(x), inplace=True)
        ], 1)


class SqueezeNet3D(nn.Module):
    def __init__(self, in_channels=3, out_dim=128):
        super(SqueezeNet3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=(1, 0, 0), ceil_mode=True),
            FireModule3D(64, 16, 64, 64),
            FireModule3D(128, 16, 64, 64),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=(1, 0, 0), ceil_mode=True),
            FireModule3D(128, 32, 128, 128),
            FireModule3D(256, 32, 128, 128),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=(1, 0, 0), ceil_mode=True),
            FireModule3D(256, 48, 192, 192),
            FireModule3D(384, 48, 192, 192),
            FireModule3D(384, 64, 256, 256),
            FireModule3D(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv3d(512, out_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.flatten(x)
        return x


class TDCNet(nn.Module):
    def __init__(
            self,
            cnn_features: int,
            num_features: int,
            output_dim=1,
            cnn_model=None,
            mlp_attention=True,
            prior_func=None,
    ):
        super().__init__()

        self.mlp_attention = mlp_attention
        self.cnn_model = cnn_model
        self.prior_func = prior_func

        # FNN layer
        self.feat_fc = nn.Linear(num_features, 32)
        self.hidden_fc = nn.Linear(32, 128)

        self.attention_layer = AttentionLayer(128 + cnn_features)
        self.output_fc = nn.Linear(128 + cnn_features, output_dim)

    def forward(self, x, features):
        out = F.dropout(F.mish(self.feat_fc(features)))
        out = F.dropout(F.mish(self.hidden_fc(out)))

        feat_out = self.cnn_model(x)
        out = torch.cat((out, feat_out), dim=1)
        if self.mlp_attention:
            out = self.attention_layer(out)
        else:
            out = out
        out = self.output_fc(out)

        if self.prior_func is not None:
            out = self.prior_func(out)
        return out


def vgg11(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = VGG3D([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512], in_channels, 1000)
    return TDCNet(1000, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def vgg13(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = VGG3D([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512], in_channels, 1000)
    return TDCNet(1000, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def vgg16(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = VGG3D([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
                      in_channels, 1000)
    return TDCNet(1000, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def vgg19(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = VGG3D(
        [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
        in_channels, 1000)
    return TDCNet(1000, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def squeezed_net(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = SqueezeNet3D(in_channels)
    return TDCNet(128, num_features, output_dim, cnn_model, prior_func=prior_func)


def resnet18(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = ResNet3D(BasicBlock3D, num_blocks=[3, 4, 6, 3], in_channels=in_channels)
    return TDCNet(512 * BasicBlock3D.expansion, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def resnet34(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = ResNet3D(BasicBlock3D, num_blocks=[3, 4, 14, 3], in_channels=in_channels)
    return TDCNet(512 * BasicBlock3D.expansion, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def resnet50(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = ResNet3D(Bottleneck3D, num_blocks=[3, 4, 6, 3], in_channels=in_channels)
    return TDCNet(512 * Bottleneck3D.expansion, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def senet50(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = ResNet3D(Bottleneck3D, num_blocks=[3, 4, 6, 3], in_channels=in_channels, channel_attention=True)
    return TDCNet(512 * Bottleneck3D.expansion, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def resnet101(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = ResNet3D(Bottleneck3D, num_blocks=[3, 4, 23, 3], in_channels=in_channels)
    return TDCNet(512 * Bottleneck3D.expansion, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def senet101(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = ResNet3D(Bottleneck3D, num_blocks=[3, 4, 23, 3], in_channels=in_channels, channel_attention=True)
    return TDCNet(512 * Bottleneck3D.expansion, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def resnet152(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = ResNet3D(Bottleneck3D, num_blocks=[3, 4, 40, 3], in_channels=in_channels)
    return TDCNet(512 * Bottleneck3D.expansion, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


def senet152(
        num_features: int,
        in_channels: int = 3,
        output_dim=1,
        prior_func: Callable = None,
        **kwargs
):
    cnn_model = ResNet3D(Bottleneck3D, num_blocks=[3, 4, 40, 3], in_channels=in_channels, channel_attention=True)
    return TDCNet(512 * Bottleneck3D.expansion, num_features, output_dim, cnn_model, prior_func=prior_func, **kwargs)


if __name__ == '__main__':
    from base_model import initialize_weights
    from utils.training_utils import train_and_eval
    from utils.data_utils import get_dataloader_v1

    # Prior 0 <= x <= 10, but need to avoid node dead
    model = squeezed_net(
        num_features=8,
        prior_func=F.leaky_relu
    )
    initialize_weights(model)
    train_loader, val_loader, test_loader = get_dataloader_v1(
        '../datasets/c2db.db',
        save_path='../datasets/hse_set_augment',
        batch_size=16,
        # select={},
        # target=['results-asr.gs.json', 'kwargs', 'data', 'gap_nosoc']
        select={'has_asr_hse': True},
        target=['results-asr.hse.json', 'kwargs', 'data', 'gap_hse_nosoc']
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=4, eta_min=0)
    criterion = nn.MSELoss()

    train_and_eval(model, train_loader, val_loader, criterion, optimizer, scheduler=scheduler,
                   checkpoint_path='../checkpoints', start_epoch=1, num_epochs=450, checkpoint_step=10)
