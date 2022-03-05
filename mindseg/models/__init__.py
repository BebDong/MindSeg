# coding=utf-8

from .canetv2 import *
from .eprnet import *

_nets_map = {
    'eprnet': EPRNet,
    'canetv2': CANetv2,
}


def get_model_by_name(name: str, **kwargs):
    return _nets_map[name.lower()](**kwargs)
