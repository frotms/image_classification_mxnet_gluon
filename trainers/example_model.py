# coding=utf-8
import os
import math
from collections import OrderedDict
import numpy as np
import mxnet as mx
from mxnet.gluon.model_zoo import vision
from utils import utils
from trainers.base_model import BaseModel
from nets.net_interface import NetModule

class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.config = config
        self.GPU_COUNT = utils.gpus_str_to_number(self.config['gpu_id'])
        self.ctx = [mx.gpu(i) for i in range(self.GPU_COUNT)]
        self.interface = NetModule(self.config['model_module_name'], self.config['model_net_name'])
        self.create_model()


    def create_model(self):
        self.net = self.interface.create_model(classes=102)
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)

    def load(self):
        # train_mode: 0:from scratch, 1:finetuning, 2:update
        # if not update all parameters:
        # for param in list(self.net.parameters())[:-1]:    # only update parameters of last layer
        #    param.requires_grad = False
        train_mode = self.config['train_mode']

        if train_mode == 'fromscratch':
            print('from scratch...')

        elif train_mode == 'finetune':
            net = self._load()
            self.net.features = net.features
            self.net.collect_params().reset_ctx(self.ctx)
            print('finetuning...')

        elif train_mode == 'update':
            _path = os.path.join(self.config['pretrained_path'], self.config['pretrained_file'])
            self.net.load_parameters(_path, ctx=self.ctx)
            self.net.collect_params().reset_ctx(self.ctx)
            print('updating...')

        else:
            ValueError('train_mode is error...')


    def _load(self):
        if self.config['model_auto_download']:
            net = self.interface.create_model(pretrained=True)
        else:
            net = self.interface.create_model()
            _path = os.path.join(self.config['pretrained_path'], self.config['pretrained_file'])
            net.load_parameters(_path, ctx=self.ctx)
        return net
