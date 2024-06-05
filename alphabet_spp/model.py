import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from setting import ALL_CHAR_SET_LEN
from captcha_setting import MAX_CAPTCHA


# 构建SPP层(空间金字塔池化层)
# Building SPP layer
class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        x_flatten = 0
        num, c, h, w = x.size()
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            # stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0] * level - h + 1) / 2),
                       math.floor((kernel_size[1] * level - w + 1) / 2))

            # update input data_n with padding
            zero_pad = torch.nn.ZeroPad2d((pooling[1], pooling[1], pooling[0], pooling[0]))
            x_new = zero_pad(x)

            # update kernel and stride
            h_new = 2 * pooling[0] + h
            w_new = 2 * pooling[1] + w

            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = f.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            else:
                tensor = f.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            # 展开、拼接
            if i == 0:
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


class CNN(nn.Module):
    def __init__(self, spp_level=3):
        super(CNN, self).__init__()
        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += (i + 1) ** 2
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
            nn.Linear(self.num_grids * 64, 1024),
            nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU())
        self.rfc = nn.Sequential(
            nn.Linear(1024, MAX_CAPTCHA*ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.spp_layer(out)
        # out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out


