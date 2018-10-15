# coding=utf-8
import time
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
mx.random.seed(1)
from trainers.base_trainer import BaseTrainer
from utils import utils

smoothing_constant =  .01
class ExampleTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, config, logger):
        super(ExampleTrainer, self).__init__(model, train_loader, val_loader, config, logger)
        self.create_optimization()
        self.GPU_COUNT = utils.gpus_str_to_number(self.config['gpu_id'])
        self.ctx = [mx.gpu(i) for i in range(self.GPU_COUNT)]

    def train_epoch(self):
        """
        training in a epoch
        :return: 
        """
        # Learning rate adjustment
        self.learning_rate = self.adjust_learning_rate(self.cur_epoch)
        self.train_batch_losses = utils.AverageMeter()
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            data = gluon.utils.split_and_load(batch_x, ctx_list=self.ctx, even_split=False)
            label = gluon.utils.split_and_load(batch_y, ctx_list=self.ctx, even_split=False)
            self.train_step(data, label)

            # printer
            self.logger.log_printer.iter_case_print(self.cur_epoch,
                                                    self.train_accuracy,self.val_accuracy,
                                                    len(self.train_loader), batch_idx+1,
                                                    self.train_batch_losses.avg, self.learning_rate)

            # mxboard summary
            if self.config['is_mxboard']:
                self.logger.summarizer.data_summarize(batch_idx, summarizer="train", summaries_dict={"lr":self.learning_rate, 'train_batch_loss':self.train_batch_losses.avg})

        time.sleep(1)


    def train_step(self, images, labels):
        """
        training in a step
        :param images: 
        :param labels: 
        :return: 
        """
        with autograd.record():
            loss = [self.get_loss(self.model.net(X), Y) for X, Y in zip(images, labels)]
        for l in loss:
            l.backward()
        self.trainer.step(self.config['batch_size'])
        curr_loss = 0.
        for l in loss:
            curr_loss += nd.sum(l).asscalar()
        curr_loss = curr_loss / self.config['batch_size']
        self.train_batch_losses.update(curr_loss, self.config['batch_size'])


    def get_loss(self, pred, label):
        """
        compute loss
        :param pred: 
        :param label: 
        :return: 
        """
        self.loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
        return self.loss_func(pred, label)


    def create_optimization(self):
        """
        optimizer
        :return: 
        """
        self.trainer = gluon.Trainer(self.model.net.collect_params(), 'sgd', {'learning_rate': self.config['learning_rate']})


    def adjust_learning_rate(self, epoch):
        """
        decay learning rate
        :param optimizer: 
        :param epoch: the first epoch is 1
        :return: 
        """
        # """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
        # lr = lr_init
        # if epoch >= num_epochs * 0.75:
        #     lr *= decay_rate ** 2
        # elif epoch >= num_epochs * 0.5:
        #     lr *= decay_rate
        learning_rate = self.config['learning_rate'] * (self.config['learning_rate_decay'] ** ((epoch - 1) // self.config['learning_rate_decay_epoch']))
        self.trainer.set_learning_rate(learning_rate)
        return learning_rate


    def evaluate_accuracy(self, data_iterator, net):
        """
        compute top-1 accuracy
        :param data_iterator: 
        :param net: 
        :return: 
        """
        loss = utils.AverageMeter()
        acc = mx.metric.Accuracy()
        for idx, (d, l) in enumerate(data_iterator):
            data = d.as_in_context(self.ctx[0])
            label = l.as_in_context(self.ctx[0])
            output = net(data)
            _loss = self.get_loss(output, label)
            curr_loss = nd.mean(_loss).asscalar()
            loss.update(curr_loss, data.shape[0])
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
            utils.view_bar(idx + 1, len(data_iterator))   # view_bar
        return acc.get()[1], loss.avg


    def evaluate_epoch(self):
        """
        evaluating in a epoch
        :return: 
        """
        # train acc
        self.train_accuracy, self.train_loss = self.evaluate_accuracy(self.train_loader, self.model.net)
        self.logger.log_printer.enter_print()
        self.logger.log_printer.str_print('train data evaluated done.')
        # test acc
        self.val_accuracy, self.val_loss = self.evaluate_accuracy(self.val_loader, self.model.net)
        self.logger.log_printer.enter_print()
        self.logger.log_printer.str_print('validation data evaluated done.')