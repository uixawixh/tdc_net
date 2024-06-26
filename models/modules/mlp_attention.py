import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, reduction=2):
        super().__init__()

        self.attention_network = nn.Sequential(
            nn.Linear(input_dim, int(input_dim / reduction)),
            nn.ReLU(inplace=True),
            nn.Linear(int(input_dim / reduction), input_dim),
            nn.Sigmoid()
        )

    def forward(self, X):
        attention_weights = self.attention_network(X)
        attention_applied = attention_weights * X
        return attention_applied
