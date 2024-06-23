from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X):
        b, c, _, _, _ = X.size()
        y = self.avg_pool(X).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return X * y.expand_as(X)
