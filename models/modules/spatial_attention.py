import torch
from torch import nn


class SpatialAttention3D(nn.Module):
    def __init__(self, out_channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
        self.conv3d = nn.Conv3d(2, out_channels, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv3d(x)
        return self.sigmoid(x)
