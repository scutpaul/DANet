#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : loss.py
import torch
def mask_iou(pred, target):

    """
    param: pred of size [N x H x W]
    param: target of size [N x H x W]
    """

    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)

    inter = torch.min(pred, target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    iou = torch.sum(inter / union) / N

    return iou

def binary_entropy_loss(pred, target, eps=0.001):

    ce = - 1.0 * target * torch.log(pred + eps) - (1 - target) * torch.log(1 - pred + eps)

    loss = torch.mean(ce)

    # TODO: training with bootstrapping

    return loss

def cross_entropy_loss(pred, target, bootstrap=0.4, eps=0.001,weight=1):


    # pred: [N x F x H x W]
    # mask: [N x F x H x W] one-hot encoded
    N, F, H, W = target.shape

    # pred = -1 * torch.log(pred + eps)
    loss = - 1.0 * target * torch.log(pred + eps) - weight * (1 - target) * torch.log(1 - pred + eps)
    # loss = - 1.0 * target * torch.log(pred + eps)
    # loss = torch.sum(pred[:, :num_object+1] * mask[:, :num_object+1])
    # loss = loss / (H * W * N)

    # bootstrap
    num = int(H * W * bootstrap)
    # print((pred*mask).shape)

    # loss = (pred* mask).view(N, F, -1)
    # print(loss.shape)
    if bootstrap == 1:
        return torch.mean(loss)
    loss = loss.view(N,F,-1)
    mloss, _ = torch.sort(loss, dim=-1, descending=True)
    loss = torch.mean(mloss[:,:, :num])

    return loss

def mask_iou_loss(pred, mask):

    N, F, H, W = mask.shape
    loss = torch.zeros(1).to(pred.device)

    for i in range(N):
        loss += (1.0 - mask_iou(pred[i], mask[i]))

    loss = loss / N
    return loss