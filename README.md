# image-classification-mxnet
This repo is designed for those who want to start their projects of image classification.
It provides fast experiment setup and attempts to maximize the number of projects killed within the given time.
It includes a few Convolutional Neural Network modules.You can build your own dnn easily.

## Requirements
Python3 support only. Tested on CUDA9.0, cudnn7.

* albumentations==0.1.1
* easydict==1.8
* imgaug==0.2.6
* opencv-python==3.4.3.18
* protobuf==3.6.1
* scikit-image==0.14.0
* mxboard         0.1.0      
* mxnet-cu90      1.3.0.post0

## model
| net                     | inputsize |
|-------------------------|-----------|
| vggnet                  | 224       |
| alexnet                 | 224       |
| resnet                  | 224       |
| inceptionV3             | 299       |
| squeezenet              | 224       |
| densenet                | 224       |
| mobilenet               | 224       |
| nasnet                  | 331       |
| resnext                 | 224       |
| senet                   | 224       |
| se_resnet               | 224       |
| squeezenet              | 224       |
| ...                     | ...       |

### pre-trained model
you can download pretrain model with url in ($net-module.py)  
And the accuracy of ImageNet pre-trained models is illustrated in the following URLs:  

- [incubator-mxnet model zoo](http://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html)  
- [classification of Gluoncv model zoo](https://gluon-cv.mxnet.io/model_zoo/classification.html)  

#### From [mxnet-gluon-vision](https://github.com/apache/incubator-mxnet/tree/master/python/mxnet/gluon/model_zoo/vision/) package:

- ResNet (`resnet18_v1`, `resnet34_v1`, `resnet50_v1`, `resnet101_v1`, `resnet152_v1`, `resnet18_v2`, `resnet34_v2`, `resnet50_v2`, `resnet101_v2`, `resnet152_v2`)
- DenseNet (`densenet121`, `densenet169`, `densenet201`, `densenet161`)
- Inception v3 (`inception_v3`)
- VGG (`vgg11`, `vgg11_bn`, `vgg13`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19`, `vgg19_bn`)
- SqueezeNet (`squeezenet1_0`, `squeezenet1_1`)
- AlexNet (`alexnet`)
- Mobilenet(`mobilenet1_0`, `mobilenet_v2_1_0`, `mobilenet0_75`, `mobilenet_v2_0_75`, `mobilenet0_5`, `mobilenet_v2_0_5`, `mobilenet0_25`, `mobilenet_v2_0_25`)

#### From [Pretrained models for Mxnet-gluon](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo/) package:
- ResNeXt (`resnext50_32x4d`, `resnext101_32x4d`, `resnext101_64x4d`, `se_resnext50_32x4d`, `se_resnext101_32x4d`, `se_resnext101_64x4d`)
- NASNet (`nasnet_4_1056`, `nasnet_5_1538`, `nasnet_7_1920`, `nasnet_6_4032`)
- SENet (`senet_52`, `senet_103`, `senet_154`)
- SE_ResNeXt (`se_resnet18_v1`, `se_resnet34_v1`, `se_resnet50_v1`, `se_resnet101_v1`, `se_resnet152_v1`, `se_resnet18_v2`, `se_resnet34_v2`, `se_resnet50_v2`, `se_resnet101_v2`, `se_resnet152_v2`)

## usage

### configuration
| configure                       | description                                                               |
|---------------------------------|---------------------------------------------------------------------------|
| model_module_name               | eg: vgg_module                                                            |
| model_net_name                  | net function name in module, eg:vgg16                                     |
| gpu_id                          | eg: single GPU: "0", multi-GPUs:"0,1,3,4,7"                               |
| is_mxboard                      | if use tensorboard for visualization                                      |
| evaluate_before_train           | evaluate accuracy before training                                         |
| shuffle                         | shuffle your training data                                                |
| data_aug                        | augment your training data                                                |
| model_auto_download             | "True" for download pretrained model with gluonAPI in a automatic manner  |
| img_height                      | input height                                                              |
| img_width                       | input width                                                               |
| num_channels                    | input channel                                                             |
| num_classes                     | output number of classes                                                  |
| batch_size                      | train batch size                                                          |
| dataloader_workers              | number of workers when loading data                                       |
| learning_rate                   | learning rate                                                             |
| learning_rate_decay             | learning rate decat rate                                                  |
| learning_rate_decay_epoch       | learning rate decay per n-epoch                                           |
| train_mode                      | eg:  "fromscratch","finetune","update"                                    |
| file_label_separator            | separator between data-name and label. eg:"----"                          |
| pretrained_path                 | pretrain model path                                                       |
| pretrained_file                 | pretrain model name. eg:"alexnet-owt-4df8aa71.pth"                        |
| pretrained_model_num_classes    | output number of classes when pretrain model trained. eg:1000 in imagenet |
| save_path                       | model path when saving                                                    |
| save_name                       | model name when saving                                                    |
| train_data_root_dir             | training data root dir                                                    |
| val_data_root_dir               | testing data root dir                                                     |
| train_data_file                 | a txt filename which has training data and label list                     |
| val_data_file                   | a txt filename which has testing data and label list                      |

### Training
1.make your training &. testing data and label list with txt file:

txt file with single label index eg:

	apple.jpg----0
	k.jpg----3
	30.jpg----0
	data/2.jpg----1
	abc.jpg----1
2.configuration

3.train

	python3 train.py

### Inference
	python3 inference.py --image images/image_00002.jpg --module vgg --net vgg16 --model /hdd/datasets/flower_102/save_model/model.params --size 224 --cls 102

### mxboard

	tensorboard --logdir=./logs/ 

logdir is log dir in your project dir 

### experiment  
There is integrated with the project using mxboard library which porved to be very useful as there is no official visualization library in mxnet. There is the learning curves for the flower-102 dataset experiment(top-1 acc: 87.74% for simple experiment).  
![](https://i.imgur.com/bMdZxpi.jpg)
![](https://i.imgur.com/xMJg8t1.jpg)  

![](https://i.imgur.com/oXn3Ovp.jpg)  
top-5:  
passion flower: 100.0%  
king protea: 3.3845638097718123e-09%  
barbeton daisy: 3.775002879735645e-10%  
purple coneflower: 1.621601507587056e-10%  
spring crocus: 1.3393154857724299e-10%    

## References
1.[http://mxnet.incubator.apache.org/](http://mxnet.incubator.apache.org/)  
2.[https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo](https://github.com/dmlc/gluon-cv/tree/master/gluoncv/model_zoo)  
3.[https://github.com/chinakook/Awesome-MXNet](https://github.com/chinakook/Awesome-MXNet)  
4.[https://github.com/awslabs/mxboard](https://github.com/awslabs/mxboard)  
5.[http://www.robots.ox.ac.uk/~vgg/data/flowers/102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102)    
6.[https://github.com/victoresque/pytorch-template](https://github.com/victoresque/pytorch-template)   