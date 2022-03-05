# coding=utf-8
# Copied from:
# https://github.com/mindspore-ai/models/blob/master/official/cv/deeplabv3/src/nets/deeplab_v3/deeplab_v3.py

from mindspore import nn
from mindspore.ops import operations as P

__all__ = ['resnet', 'Resnet']


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     weight_init='xavier_uniform')


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad',
                     padding=padding, dilation=dilation, weight_init='xavier_uniform')


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1,
                 use_batch_statistics=True):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)
        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  use_batch_statistics=use_batch_statistics)

        self.relu = nn.ReLU()
        self.downsample = downsample

        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity)
        out = self.relu(out)
        return out


class Resnet(nn.Cell):
    def __init__(self, block, block_num, output_stride, use_batch_statistics=True):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, pad_mode='pad',
                               padding=3, weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(self.inplanes, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, block_num[0],
                                       use_batch_statistics=use_batch_statistics)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2,
                                       use_batch_statistics=use_batch_statistics)

        if output_stride == 16:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1,
                                           base_dilation=2, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)
        elif output_stride == 8:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=1, base_dilation=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1,
                                           base_dilation=4, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)

    def _make_layer(self, block, planes, blocks, stride=1, base_dilation=1, grids=None,
                    use_batch_statistics=True):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=use_batch_statistics)
            ])

        if grids is None:
            grids = [1] * blocks

        layers = [
            block(self.inplanes, planes, stride, downsample, dilation=base_dilation * grids[0],
                  use_batch_statistics=use_batch_statistics)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=base_dilation * grids[i],
                      use_batch_statistics=use_batch_statistics))

        return nn.SequentialCell(layers)

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        c1 = self.layer1(out)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


def resnet(layers, **kwargs):
    assert layers in (50, 101, 152)
    if layers == 50:
        model = Resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif layers == 101:
        model = Resnet(Bottleneck, [3, 4, 23, 3], **kwargs)
    else:
        model = Resnet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
