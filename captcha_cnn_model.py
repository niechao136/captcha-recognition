import torch
import torch.nn as nn
import torch.nn.functional as f
from captcha_setting import ALL_CHAR_SET_LEN, MAX_CAPTCHA


class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = f.max_pool2d(x, kernel_size=kernel_size, stride=kernel_size).view(bs, -1)
            else:
                tensor = f.avg_pool2d(x, kernel_size=kernel_size, stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x


class CaptchaCNN(nn.Module):

    def __init__(self, spp_level=3):
        super(CaptchaCNN, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2 ** (i * 2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.spp_layer = SPPLayer(spp_level)

        self.fc = nn.Sequential(
            nn.Linear((160 // 8) * (60 // 8) * 64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, MAX_CAPTCHA * ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.spp_layer(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out
