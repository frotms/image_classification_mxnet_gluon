# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import dataset
from data_loader.data_processor import DataProcessor


class MxnetDataset(dataset.Dataset):
    def __init__(self, txt, config, transform=None, is_train_set=True):
        self.config = config
        imgs = []
        # load image filename list here
        with open(txt, 'r') as f:
            for line in f:
                line = line.strip('\n\r').strip('\n').strip('\r')
                words = line.split(self.config['file_label_separator'])
                # single label here so we use int(words[1])
                imgs.append((words[0], int(words[1])))
        self.DataProcessor = DataProcessor(self.config)
        self.imgs = imgs
        self.transform = transform
        self.is_train_set = is_train_set

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # cv2.imread
        _root_dir = self.config['train_data_root_dir'] if self.is_train_set else self.config['val_data_root_dir']
        image = self.self_defined_loader(os.path.join(_root_dir, fn))
        image = np.transpose(image, (2, 0, 1))
        # ndarray to mx_nd_array
        image = mx.nd.array(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, float(label)


    def __len__(self):
        return len(self.imgs)


    def self_defined_loader(self, filename):
        image = self.DataProcessor.image_loader(filename)
        image = self.DataProcessor.image_resize(image)
        if self.is_train_set and self.config['data_aug']:
            image = self.DataProcessor.data_aug(image)
        image = self.DataProcessor.input_norm(image)
        return image


def get_data_loader(config):
    train_data_file = config['train_data_file']
    test_data_file = config['val_data_file']
    batch_size = config['batch_size']
    num_workers =config['dataloader_workers']
    shuffle = config['shuffle']

    if not os.path.isfile(train_data_file):
        raise ValueError('train_data_file is not existed')
    if not os.path.isfile(test_data_file):
        raise ValueError('val_data_file is not existed')

    train_data = MxnetDataset(txt=train_data_file,config=config,
                           transform=None, is_train_set=True)
    test_data = MxnetDataset(txt=test_data_file,config=config,
                                transform=None, is_train_set=False)
    train_loader = gluon.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers)
    test_loader = gluon.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, test_loader
