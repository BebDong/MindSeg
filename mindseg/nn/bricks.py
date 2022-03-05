# coding=utf-8

from mindspore import nn
from mindspore.ops import operations as P

__all__ = ['ConvModule2d', 'GlobalFlow']


class ConvModule2d(nn.Cell):
    def __init__(self, in_planes, planes, kernel_size, stride=1, pad_mode='pad', padding=0,
                 dilation=1, group=1, has_bias=False, weight_init='xavier_uniform',
                 has_bn=True, use_batch_statistics=True, activation='relu', **kwargs):
        super(ConvModule2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size, stride, pad_mode, padding,
                              dilation, group, has_bias, weight_init, **kwargs)
        self.bn = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics) if has_bn else None
        self.act = nn.get_activation(activation) if activation else None

    def construct(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.act:
            x = self.act(x)
        return x


class GlobalFlow(nn.Cell):
    def __init__(self, in_planes, planes, use_batch_statistics=True):
        super(GlobalFlow, self).__init__()
        self.conv = ConvModule2d(in_planes, planes, 1, weight_init='xavier_uniform',
                                 use_batch_statistics=use_batch_statistics)
        self.shape = P.Shape()

    def construct(self, x):
        size = self.shape(x)
        out = nn.AvgPool2d(size[2])(x)
        out = self.conv(out)
        out = P.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        # out = P.ResizeBilinear((size[2], size[3]), True)(out)  # do not support GPU yet
        return out
