#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : Logger.py
import torch
import numpy as np
import time
import sys


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def avg(self):
        return self.sum / self.count


class Loss_record():
    '''save the loss: total(tensor), part1 and part2 can be 0'''

    def __init__(self):
        self.total = AverageMeter()
        self.part1 = AverageMeter()
        self.part2 = AverageMeter()
        self.part3 = AverageMeter()
        self.part4 = AverageMeter()

    def reset(self):
        self.total.reset()
        self.part1.reset()
        self.part2.reset()
        self.part3.reset()
        self.part4.reset()

    def updateloss(self, loss_val, loss_part1=0, loss_part2=0, loss_part3=0, loss_part4=0):
        self.total.update(loss_val.data.item(), 1)
        self.part1.update(loss_part1.data.item(), 1) if isinstance(loss_part1, torch.Tensor) else self.part1.update(0,
                                                                                                                    1)
        self.part2.update(loss_part2.data.item(), 1) if isinstance(loss_part2, torch.Tensor) else self.part2.update(0,
                                                                                                                    1)
        self.part3.update(loss_part3.data.item(), 1) if isinstance(loss_part3, torch.Tensor) else self.part3.update(0,
                                                                                                                    1)
        self.part4.update(loss_part4.data.item(), 1) if isinstance(loss_part4, torch.Tensor) else self.part4.update(0,
                                                                                                                    1)

    def getloss(self, epoch, step):
        ''' get every step loss and reset '''
        total_avg = self.total.avg()
        part1_avg = self.part1.avg()
        part2_avg = self.part2.avg()
        part3_avg = self.part3.avg()
        part4_avg = self.part4.avg()
        out_str = 'epoch %d, step %d : %.4f, %.4f, %.4f, %.4f, %.4f' % \
                  (epoch, step, total_avg, part1_avg, part2_avg, part3_avg, part4_avg)
        self.reset()
        return out_str



def measure(y_in, pred_in):
    thresh = .5
    y = y_in > thresh
    pred = pred_in > thresh
    tp = np.logical_and(y, pred).sum()
    tn = np.logical_and(np.logical_not(y), np.logical_not(pred)).sum()
    fp = np.logical_and(np.logical_not(y), pred).sum()
    fn = np.logical_and(y, np.logical_not(pred)).sum()
    return tp, tn, fp, fn


from libs.utils.davis_JF import db_eval_boundary, db_eval_iou


class TreeEvaluation():
    '''eval training output'''

    def __init__(self, class_list=None):
        assert class_list is not None
        self.class_indexes = class_list
        self.num_classes = len(class_list)
        self.setup()

    def setup(self):
        self.tp_list = [0] * self.num_classes
        self.f_list = [0] * self.num_classes
        self.j_list = [0] * self.num_classes
        self.n_list = [0] * self.num_classes
        self.total_list = [0] * self.num_classes
        self.iou_list = [0] * self.num_classes

        self.f_score = [0] * self.num_classes
        self.j_score = [0] * self.num_classes

    def update_evl(self, idx, query_mask, pred):
        # B N H W
        batch = len(idx)
        for i in range(batch):
            if not isinstance(idx[i], int):
                id = idx[i].item()
            else:
                id = idx[i]
            id = self.class_indexes.index(id)
            tp, total = self.test_in_train(query_mask[i], pred[i])
            for j in range(query_mask[i].shape[0]):
                thresh = .5
                y = query_mask[i][j].cpu().numpy() > thresh
                predict = pred[i][j].data.cpu().numpy() > thresh
                self.f_list[id] += db_eval_boundary(predict, y)
                self.j_list[id] += db_eval_iou(y, predict)
                self.n_list[id] += 1

            self.tp_list[id] += tp
            self.total_list[id] += total
        self.iou_list = [self.tp_list[ic] /
                         float(max(self.total_list[ic], 1))
                         for ic in range(self.num_classes)]
        self.f_score = [self.f_list[ic] /
                        float(max(self.n_list[ic], 1))
                        for ic in range(self.num_classes)]
        self.j_score = [self.j_list[ic] /
                        float(max(self.n_list[ic], 1))
                        for ic in range(self.num_classes)]

    def test_in_train(self, query_label, pred):
        # test N*H*F
        pred = pred.data.cpu().numpy()
        query_label = query_label.cpu().numpy()

        tp, tn, fp, fn = measure(query_label, pred)
        total = tp + fp + fn
        return tp, total

    def logiou(self, epoch=None, step=None):
        mean_iou = np.mean(self.iou_list)
        out_str = 'iou: %.4f' % mean_iou
        self.setup()
        return out_str



class TimeRecord():
    def __init__(self, maxstep, max_epoch):
        self.maxstep = maxstep
        self.max_epoch = max_epoch

    def gettime(self, epoch, begin_time):
        step_time = time.time() - begin_time
        remaining_time = (self.max_epoch - epoch) * step_time * self.maxstep / 3600
        return step_time, remaining_time


class LogTime():
    def __init__(self):
        self.reset()

    def t1(self):
        self.logt1 = time.time()

    def t2(self):
        self.logt2 = time.time()
        self.alltime += (self.logt2 - self.logt1)

    def reset(self):
        self.logt1 = None
        self.logt2 = None
        self.alltime = 0

    def getalltime(self):
        out = self.alltime
        self.reset()
        return out