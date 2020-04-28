
import torch.nn as nn
from torch.nn.utils import weight_norm
import math


def required_layers_num(window_size):
    # receptive_field = 2^0 + 2^1 + ... + 2^(layers_num - 1) = 2^(layers_num) - 1 = window_size - 1
    return int(math.log2(window_size)) + 1


def receptive_field(layers_num):
    # receptive_field = 2^0 + 2^1 + ... + 2^(layers_num - 1) = 2^(layers_num) - 1
    return (2 ** layers_num) - 1


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class NonCausalTCN(nn.Module):
    def __init__(self, event_size, channels, dropout=0.2):
        super(NonCausalTCN, self).__init__()
        kernel_size = 3
        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = event_size if i == 0 else channels[i - 1]
            out_channels = channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
