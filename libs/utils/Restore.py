#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : Restore.py
import os
import shutil
import torch
import numpy as np

def restore(args, model, test_best=False):

    group = args.group
    savedir = args.snapshot_dir
    filename='epoch_%d.pth.tar'%(args.restore_epoch)
    if test_best:
        filename='model_best.pth.tar'
    snapshot = os.path.join(savedir, filename)
    assert os.path.exists(snapshot), "Snapshot file %s does not exist."%(snapshot)

    checkpoint = torch.load(snapshot)
    weight = checkpoint['state_dict']
    s = model.state_dict()
    for key, val in weight.items():

        # process ckpt from parallel module
        if key[:6] == 'module':
            key = key[7:]

        if key in s and s[key].shape == val.shape:
            s[key][...] = val
        elif key not in s:
            print('ignore weight from not found key {}'.format(key))
        else:
            print('ignore weight of mistached shape in key {}'.format(key))
    model.load_state_dict(s)

    print('Loaded weights from %s'%(snapshot))
    return weight

def restore_from_weight(model,weight):
    s = model.state_dict()
    for key, val in weight.items():

        # process ckpt from parallel module
        if key[:6] == 'module':
            key = key[7:]

        if key in s and s[key].shape == val.shape:
            s[key][...] = val
        elif key not in s:
            print('ignore weight from not found key {}'.format(key))
        else:
            print('ignore weight of mistached shape in key {}'.format(key))
    model.load_state_dict(s)


def get_model_para_number(model):
    total_number = 0
    for para in model.parameters():
        total_number += torch.numel(para)

    return total_number

def get_save_dir(args):
    snapshot_dir = os.path.join(args.snapshot_dir, args.arch, 'id_%d_group_%d_of_%d'%(args.trainid, args.group, args.num_folds))
    return snapshot_dir

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    # savedir = os.path.join(args.snapshot_dir, args.arch, 'id_%d_group_%d_of_%d'%(args.trainid, args.group, args.num_folds))
    savedir = args.snapshot_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savepath = os.path.join(savedir, filename)
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(savedir, 'model_best.pth.tar'))

def save_model(args, epoch, model, optimizer, is_best=False):
    if epoch % args.save_epoch == 0 or is_best:
        save_checkpoint(args,{'state_dict': model.state_dict()},
                        is_best=is_best,filename='epoch_%d.pth.tar'% (epoch))

