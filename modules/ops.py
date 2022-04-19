import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import functools
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _routing_two(nn.Module):
    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing_two, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)
        self.fc2 = nn.Linear(in_channels, num_experts)

    def forward(self, x1, x2):
        x1 = torch.flatten(x1)
        x1 = self.dropout(x1)
        x1 = self.fc(x1)

        x2 = torch.flatten(x2)
        x2 = self.dropout(x2)
        x2 = self.fc2(x2)
        return torch.sigmoid(x1 + x2)


class CDM(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
        num_experts=3,
        dropout_rate=0.2,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CDM, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
        )

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._max_pooling = functools.partial(F.adaptive_max_pool2d, output_size=(1, 1))
        self._routing_fn_two = _routing_two(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, *kernel_size)
        )

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(
                F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                weight,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, inputs, attn_inputs=None):
        b, _, _, _ = inputs.size()
        res = []
        assert attn_inputs != None
        for input, attn_input in zip(inputs, attn_inputs):
            input = input.unsqueeze(0)
            attn_input = attn_input.unsqueeze(0)
            avg_pooled_inputs = self._avg_pooling(attn_input)
            max_pooled_inputs = self._max_pooling(attn_input)
            routing_weights = self._routing_fn_two(avg_pooled_inputs, max_pooled_inputs)
            kernels = torch.sum(
                routing_weights[:, None, None, None, None] * self.weight, 0
            )
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)


def model_structure(model):
    blank = ' '
    # print('-' * 90)
    # print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|'
    #       + ' ' * 15 + 'weight shape' + ' ' * 15 + '|'
    #         + ' ' * 3 + 'number' + ' ' * 3 + '|')
    # print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print(
        'The parameters of Model {}: {:4f}M'.format(
            model._get_name(), num_para * type_size / 1000 / 1000
        )
    )
    print('-' * 90)
