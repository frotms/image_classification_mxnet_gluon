# coding: utf-8
# pylint: disable= arguments-differ,missing-docstring
"""ResNext, implemented in Gluon."""
from __future__ import division

__all__ = ['ResNext', 'Block', 'get_resnext',
           'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
           'se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnext101_64x4d']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock


class Block(HybridBlock):
    r"""Bottleneck Block from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    """
    def __init__(self, channels, cardinality, bottleneck_width, stride,
                 downsample=False, last_gamma=False, use_se=False, **kwargs):
        super(Block, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(group_width, kernel_size=1, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(group_width, kernel_size=3, strides=stride, padding=1,
                                groups=cardinality, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels * 4, kernel_size=1, use_bias=False))
        if last_gamma:
            self.body.add(nn.BatchNorm())
        else:
            self.body.add(nn.BatchNorm(gamma_initializer='zeros'))

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Conv2D(channels // 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Conv2D(channels * 4, kernel_size=1, padding=0))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels * 4, kernel_size=1, strides=stride,
                                          use_bias=False))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x


# Nets
class ResNext(HybridBlock):
    r"""ResNext model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    classes : int, default 1000
        Number of classification classes.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    """
    def __init__(self, layers, cardinality, bottleneck_width,
                 classes=1000, last_gamma=False, use_se=False, **kwargs):
        super(ResNext, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        channels = 64

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels, 7, 2, 3, use_bias=False))

            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(channels, num_layer, stride,
                                                   last_gamma, use_se, i+1))
                channels *= 2
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.Dense(classes)

    def _make_layer(self, channels, num_layers, stride, last_gamma, use_se, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(Block(channels, self.cardinality, self.bottleneck_width,
                            stride, True, last_gamma=last_gamma, use_se=use_se, prefix=''))
            for _ in range(num_layers-1):
                layer.add(Block(channels, self.cardinality, self.bottleneck_width,
                                1, False, last_gamma=last_gamma, use_se=use_se, prefix=''))
        return layer

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


# Specification
resnext_spec = {50: [3, 4, 6, 3],
                101: [3, 4, 23, 3]}


# Constructor
def get_resnext(num_layers, cardinality=32, bottleneck_width=4, use_se=False,
                pretrained=False, ctx=cpu(),
                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    num_layers : int
        Numbers of layers. Options are 50, 101.
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert num_layers in resnext_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnext_spec.keys()))
    layers = resnext_spec[num_layers]
    net = ResNext(layers, cardinality, bottleneck_width, use_se=use_se, **kwargs)
    if pretrained:
        from mxnet.gluon.model_zoo.model_store import get_model_file
        if not use_se:
            net.load_params(get_model_file('resnext%d_%dx%dd'%(num_layers, cardinality,
                                                               bottleneck_width),
                                           tag=pretrained, root=root), ctx=ctx)
        else:
            net.load_params(get_model_file('se_resnext%d_%dx%dd'%(num_layers, cardinality,
                                                                  bottleneck_width),
                                           tag=pretrained, root=root), ctx=ctx)

    return net

def resnext50_32x4d(**kwargs):
    r"""ResNext50 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resnext(50, 32, 4, use_se=False, **kwargs)

def resnext101_32x4d(**kwargs):
    r"""ResNext101 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resnext(101, 32, 4, use_se=False, **kwargs)

def resnext101_64x4d(**kwargs):
    r"""ResNext101 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resnext(101, 64, 4, use_se=False, **kwargs)

def se_resnext50_32x4d(**kwargs):
    r"""SE-ResNext50 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resnext(50, 32, 4, use_se=True, **kwargs)

def se_resnext101_32x4d(**kwargs):
    r"""SE-ResNext101 32x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resnext(101, 32, 4, use_se=True, **kwargs)

def se_resnext101_64x4d(**kwargs):
    r"""SE-ResNext101 64x4d model from
    `"Aggregated Residual Transformations for Deep Neural Network"
    <http://arxiv.org/abs/1611.05431>`_ paper.
    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resnext(101, 64, 4, use_se=True, **kwargs)