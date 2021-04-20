#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : optimer.py
from torch.optim import SGD, Adam
def DAN_optimizer(model):
    for param in model.encoder.parameters():
        param.requires_grad = False

    opt = Adam(
        [
            {'params': model.support_qkv.parameters()},
            {'params': model.query_qkv.parameters()},
            {'params': model.conv_q.parameters()},
            {'params': model.Decoder.parameters()}
        ],
        lr=1e-5, betas=(0.9, 0.999), weight_decay=5e-4)

    return opt

def finetune_optimizer(model):
    for param in model.encoder.layer0.parameters():
        param.requires_grad = False
    for param in model.encoder.layer1.parameters():
        param.requires_grad = False
    for param in model.encoder.layer2.parameters():
        param.requires_grad = False

    opt = Adam(
        [
            {'params': model.encoder.layer3.parameters()},],
        lr=5e-6,  betas=(0.9, 0.999),weight_decay=5e-4)

    return opt