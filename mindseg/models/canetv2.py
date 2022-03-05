# coding=utf-8

import math
from mindspore import nn
from mindspore.ops import operations as P

from .resnet import resnet
from mindseg.nn import GlobalFlow, ConvModule2d

__all__ = ['CANetv2']


class CANetv2(nn.Cell):
    def __init__(self, phase='train', nclass=21, output_stride=8, freeze_bn=False):
        super(CANetv2, self).__init__()
        use_batch_statistics = not freeze_bn
        self.backbone = resnet(101, output_stride=output_stride,
                               use_batch_statistics=use_batch_statistics)
        self.head = _CANetHead(phase, 2048, 512, nclass, use_batch_statistics)
        self.shape = P.Shape()

    def construct(self, x):
        size = self.shape(x)
        _, _, c3, c4 = self.backbone(x)
        out = self.head(c3, c4)
        out = P.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        return out


class _CANetHead(nn.Cell):
    def __init__(self, phase, in_planes, planes, num_classes, use_batch_statistics=True):
        super(_CANetHead, self).__init__()
        self.phase = phase

        self.comp_c4 = nn.Conv2d(in_planes, planes, 1, weight_init='xavier_uniform')
        self.comp_c3 = nn.Conv2d(in_planes // 2, planes, 1, weight_init='xavier_uniform')
        self.gap = GlobalFlow(in_planes, 48, use_batch_statistics)

        self.ed1 = _EncoderDecoder(2, planes, use_batch_statistics)
        self.ed2 = _EncoderDecoder(4, planes, use_batch_statistics)
        self.ed3 = _EncoderDecoder(8, planes, use_batch_statistics)

        self.gate1 = ConvModule2d(planes * 4, planes, 1, weight_init='xavier_uniform',
                                  use_batch_statistics=use_batch_statistics)
        self.gate2 = ConvModule2d(planes, 4, 1, weight_init='xavier_uniform',
                                  use_batch_statistics=use_batch_statistics, activation=None)

        self.conv1 = ConvModule2d(planes + 48, 256, 1, weight_init='xavier_uniform',
                                  use_batch_statistics=use_batch_statistics)
        self.drop = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(256, num_classes, 1, weight_init='xavier_uniform', has_bias=True)

        self.concat = P.Concat(axis=1)
        self.softmax = P.Softmax(axis=1)
        self.expand = P.ExpandDims()
        self.mul = P.Mul()

    def construct(self, c3, c4):
        gap = self.gap(c4)
        c3 = self.comp_c3(c3)
        c4 = self.comp_c4(c4)

        mid1 = self.ed1(c4)
        mid2 = self.ed2(c4)
        mid3 = self.ed3(c4)

        score = self.concat([mid1, mid2, mid3, c3])
        score = self.gate2(self.gate1(score))
        score = self.softmax(score)
        score = self.expand(score, 2)

        c3 = self.expand(c3, 1)
        mid1 = self.expand(mid1, 1)
        mid2 = self.expand(mid2, 1)
        mid3 = self.expand(mid3, 1)
        mid = self.concat([mid1, mid2, mid3, c3])
        out = self.mul(mid, score)
        out = out.sum(axis=1, keepdims=False)

        out = self.concat([out, gap])
        out = self.conv1(out)
        if self.phase == 'train':
            out = self.drop(out)
        out = self.conv2(out)
        return out


class _EncoderDecoder(nn.Cell):
    def __init__(self, scale, planes, use_batch_statistics=True):
        super(_EncoderDecoder, self).__init__()
        num_blk = int(math.log(scale, 2))
        self.encoder = nn.SequentialCell()
        for i in range(num_blk):
            self.encoder.append(_Bottleneck(planes, use_batch_statistics))
        self.conv3x3 = ConvModule2d(planes, planes, 3, 1, padding=1, weight_init='xavier_uniform',
                                    use_batch_statistics=use_batch_statistics)
        self.decoder = ConvModule2d(planes, planes, 1, weight_init='xavier_uniform',
                                    use_batch_statistics=use_batch_statistics)
        self.shape = P.Shape()

    def construct(self, x):
        size = self.shape(x)
        out = self.encoder(x)
        out = self.conv3x3(out)
        out = P.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        out = self.decoder(out)
        return out


class _Bottleneck(nn.Cell):
    def __init__(self, planes, use_batch_statistics=True):
        super(_Bottleneck, self).__init__()
        inner_planes = planes // 4
        self.conv1 = ConvModule2d(planes, inner_planes, 1, weight_init='xavier_uniform',
                                  use_batch_statistics=use_batch_statistics)
        self.conv2 = ConvModule2d(inner_planes, inner_planes, 3, 2, padding=1,
                                  weight_init='xavier_uniform',
                                  use_batch_statistics=use_batch_statistics)
        self.conv3 = ConvModule2d(inner_planes, planes, 1, weight_init='xavier_uniform',
                                  use_batch_statistics=use_batch_statistics, activation=None)
        self.conv4 = ConvModule2d(planes, planes, 1, 2, weight_init='xavier_uniform',
                                  use_batch_statistics=use_batch_statistics, activation=None)
        self.relu = nn.ReLU()
        self.add = P.Add()

    def construct(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.add(out, self.conv4(identity))
        out = self.relu(out)
        return out
