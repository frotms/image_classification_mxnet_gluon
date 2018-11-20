#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import mxnet as mx
from importlib import import_module

class TagMxnetInference(object):

    def __init__(self, **kwargs):
        _input_size = kwargs.get('input_size',224)
        self.input_size = (_input_size, _input_size)
        self.gpu_index = kwargs.get('gpu_index', '0')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_index
        self.ctx = mx.gpu()
        self.net = self._create_model(**kwargs)
        self._load(**kwargs)
        

    def close(self):
        pass


    def _create_model(self, **kwargs):
        module_name = kwargs.get('module_name','vgg')
        net_name = kwargs.get('net_name', 'vgg16')
        m = import_module('nets.' + module_name)
        model = getattr(m, net_name)
        classes = kwargs.get('classes')
        net = model(classes=classes)
        return net


    def _load(self, **kwargs):
        model_name = kwargs.get('model_name', 'model.params')
        model_filename = model_name
        self.net.load_parameters(model_filename, ctx=self.ctx)


    def run(self, image_data, **kwargs):
        _image_data = self.image_preproces(image_data)
        input = mx.nd.array(_image_data).as_in_context(self.ctx)
        logit = self.net(input)
        # softmax
        pred = mx.ndarray.softmax(logit, axis=1)
        return pred.asnumpy()
        # return self.softmax(logit.asnumpy())


    def image_preproces(self, image_data):
        _image = cv2.resize(image_data, self.input_size)
        _image = _image[:,:,::-1]   # bgr2rgb
        _image = np.transpose(_image, (2, 0, 1))
        _image = (_image*1.0 - 127) * 0.0078125 # 1/128
        _image = np.expand_dims(_image, 0)
        return _image.astype(np.float32)


    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x) / np.sum(np.exp(x), axis=1)

if __name__ == "__main__":
    # # python3 inference.py --image images/image_00002.jpg --module vgg --net vgg16 --model model.params --size 224 --cls 102
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-image', "--image", type=str, help='Assign the image path.', default=None)
    parser.add_argument('-module', "--module", type=str, help='Assign the module name.', default=None)
    parser.add_argument('-net', "--net", type=str, help='Assign the net name.', default=None)
    parser.add_argument('-model', "--model", type=str, help='Assign the net name.', default=None)
    parser.add_argument('-cls', "--cls", type=int, help='Assign the classes number.', default=None)
    parser.add_argument('-size', "--size", type=int, help='Assign the input size.', default=None)
    args = parser.parse_args()
    if args.image is None or args.module is None or args.net is None or args.model is None\
            or args.size is None or args.cls is None:
        raise TypeError('input error')
    if not os.path.exists(args.model):
        raise TypeError('cannot find file of model')
    print('test:')
    filename = args.image
    module_name = args.module
    net_name = args.net
    model_name = args.model
    input_size = args.size
    num_classes = args.cls
    image = cv2.imread(filename)
    if image is None:
        raise TypeError('image data is none')
    tagInfer = TagMxnetInference(module_name=module_name,net_name=net_name,
                                   classes=num_classes, model_name=model_name,
                                   input_size=input_size)
    result = tagInfer.run(image)
    print(result)
    print(np.argmax(result[0]))
    print('done!')