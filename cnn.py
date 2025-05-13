import torch
from torch import nn

class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(ConvFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )

    def forward(self, x):
        return self.conv(x)
