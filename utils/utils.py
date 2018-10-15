# coding=utf-8
from __future__ import print_function
import numpy as np
import time
import sys
import os


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def view_bar(num, total):
    """
    
    :param num: 
    :param total: 
    :return: 
    """
    rate = float(num + 1) / total
    rate_num = int(rate * 100)
    if num != total:
        r = '\r[%s%s]%d%%' % ("=" * rate_num, " " * (100 - rate_num), rate_num,)
    else:
        r = '\r[%s%s]%d%%' % ("=" * 100, " " * 0, 100,)
    sys.stdout.write(r)
    sys.stdout.flush()


def gpus_str_to_number(gpus_index_str):
    """
    analysis gpus number with CUDA_VISIBLE_DEVICES
    :param gpus_index_str: 
    :return: 
    """
    _str = gpus_index_str.replace(' ','')
    _list = _str.split(',')
    gpus_number = len(_list)
    return gpus_number

