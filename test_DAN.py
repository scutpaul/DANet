#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : test_DAN.py
import argparse
import json
import os
import time

from libs.config.DAN_config import OPTION as opt
from libs.utils.Logger import TreeEvaluation as Evaluation, TimeRecord, LogTime, Tee, Loss_record
from libs.utils.Restore import get_save_dir,restore
from libs.models.DAN import *
from libs.dataset.YoutubeVOS import YTVOSDataset
from libs.dataset.transform import TestTransform
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from libs.utils.loss import *
from libs.utils.optimer import finetune_optimizer

SNAPSHOT_DIR = opt.SNAPSHOT_DIR

def get_arguments():
    parser = argparse.ArgumentParser(description='FSVOS')
    parser.add_argument("--arch", type=str,default='DAN') #
    parser.add_argument("--data_path", type=str,default=None)
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR)
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--restore_epoch", type=int, default=0)
    parser.add_argument("--query_frame", type=int, default=5)
    parser.add_argument("--support_frame", type=int, default=5)
    parser.add_argument("--finetune_idx", type=int, default=1)

    parser.add_argument("--test", action='store_true')
    parser.add_argument("--test_best", action='store_true')
    parser.add_argument("--finetune", action='store_true')
    parser.add_argument("--finetune_step", type=int, default=21)
    parser.add_argument("--finetune_valstep", type=int, default=5)
    parser.add_argument("--finetune_weight", type=float, default=0.1)
    parser.add_argument("--finetune_iou", type=float, default=0.5)
    parser.add_argument("--test_num", type=int, default=1)

    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--trainid", type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False

def finetune(args, model, imgs, masks, test_list):
    print('start finetune', args.finetune_step, args.finetune_valstep)
    B, N, C, H, W = imgs.shape
    GT = masks.squeeze(2) # B N H W
    losses = Loss_record()
    class_list = test_list
    valid_evaluations = Evaluation(class_list=class_list)
    optimizer = finetune_optimizer(model)
    celoss = cross_entropy_loss
    criterion = lambda pred, target, bootstrap=1: [celoss(pred, target, bootstrap,weight=args.finetune_weight), mask_iou_loss(pred, target)]
    stop_iou = args.finetune_iou
    pred_map = model(imgs, imgs, masks)
    pred_map = pred_map.squeeze(2)
    valid_evaluations.update_evl(class_list, GT, pred_map)
    if np.mean(valid_evaluations.j_score) > stop_iou:
        print('No need for online learning')
        return
    valid_evaluations.logiou()

    model.train()
    model.apply(fix_bn)
    for train_step in range(args.finetune_step):
        for i in range(N):
            img = imgs[:,i:i+1]
            mask = GT[:,i:i+1]
            pred_map = model(img, imgs, masks)
            pred_map = pred_map.squeeze(2)
            few_ce_loss,few_iou_loss = criterion(pred_map,mask,bootstrap=1)
            total_loss = few_ce_loss + few_iou_loss
            losses.updateloss(total_loss, few_ce_loss,few_iou_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            valid_evaluations.update_evl(class_list, mask, pred_map)
        if train_step % args.finetune_valstep == 0:
            mean_iou = np.mean(valid_evaluations.j_score)
            if mean_iou > stop_iou:
                print('stop_finetune',mean_iou)
                break
            iou_str = valid_evaluations.logiou(0, train_step)
            loss_str = losses.getloss(0, train_step)
            print(loss_str, ' | ', iou_str, ' | ')

    finetune_path = os.path.join(args.finetune_path,'model_test_num_%d.pth.tar' % args.test_num)
    torch.save(model.state_dict(), finetune_path)


def test(args):
    model = eval(args.arch).DAN()
    model.eval()
    size = opt.test_size
    tsfm_test = TestTransform(size)
    finetune_idx = None
    if args.finetune:
        finetune_idx = args.finetune_idx
    test_dataset = YTVOSDataset(data_path=opt.root_path, train=False, query_frame=args.query_frame,support_frame=args.support_frame,
                                transforms=tsfm_test, set_index=args.group, finetune_idx=finetune_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    test_list = test_dataset.get_class_list()
    model.cuda()

    print('test_group:',args.group, '  test_num:', len(test_dataloader))


    if args.test_best:
        restore(args,model,test_best=True)
        print("Resume best model...")
    if args.restore_epoch > 0:
        restore(args, model)
        print("Resume training...")
        print("Resume_epoch: %d" % (args.restore_epoch))
        args.snapshot_dir = os.path.join(args.snapshot_dir, str(args.restore_epoch))
        if not os.path.exists(args.snapshot_dir):
            os.mkdir(args.snapshot_dir)

    test_evaluations = Evaluation(class_list=test_list)

    support_img,support_mask = None,None
    for index, data in enumerate(test_dataloader):

        video_query_img, video_query_mask, new_support_img, new_support_mask, idx, vid, begin_new = data

        if begin_new:
            support_img, support_mask = new_support_img.cuda(), new_support_mask.cuda()
            if args.finetune:
                finetune(args, model, support_img, support_mask, test_list)
                model.eval()
        b, len_video, c, h, w = video_query_img.shape
        step_len = (len_video // args.query_frame)
        if len_video % args.query_frame != 0:
            step_len = step_len+1
        test_len = step_len

        for i in range(test_len):
            if i == step_len - 1:
                query_img = video_query_img[:, i*args.query_frame:]
                query_mask = video_query_mask[:, i*args.query_frame:]
            else:
                query_img = video_query_img[:, i*args.query_frame:(i+1)*args.query_frame]
                query_mask = video_query_mask[:, i*args.query_frame:(i+1)*args.query_frame]
            query_img, query_mask,  idx \
                = query_img.cuda(), query_mask.cuda(),  idx.cuda()
            with torch.no_grad():
                pred_map = model(query_img, support_img, support_mask)
            pred_map = pred_map.squeeze(2) # B N 1 H W -> B N H W
            query_mask = query_mask.squeeze(2)
            test_evaluations.update_evl(idx, query_mask, pred_map)

    mean_f = np.mean(test_evaluations.f_score)
    str_mean_f = 'F: %.4f ' % (mean_f)
    mean_j = np.mean(test_evaluations.j_score)
    str_mean_j = 'J: %.4f ' % (mean_j)

    f_list = ['%.4f' % n for n in test_evaluations.f_score]
    str_f_list = ' '.join(f_list)
    j_list = ['%.4f' % n for n in test_evaluations.j_score]
    str_j_list = ' '.join(j_list)

    print(str_mean_f, str_f_list + '\n')
    print(str_mean_j, str_j_list + '\n')

if __name__ == '__main__':
    args = get_arguments()
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    if not os.path.exists(get_save_dir(args)):
        os.makedirs(get_save_dir(args))
    args.snapshot_dir = get_save_dir(args)


    if args.finetune:
        args.finetune_path = os.path.join(args.snapshot_dir,str(args.finetune_idx),'test_'+str(args.test_num))
        if not os.path.exists(args.finetune_path):
            os.makedirs(args.finetune_path)
            logger = Tee(os.path.join(args.finetune_path, 'finetune_%d_test_%d.txt' % (args.finetune_idx, args.test_num)), 'w')
    elif args.test_best:
        logger = Tee(os.path.join(args.snapshot_dir, 'test_best_%d.txt' % args.test_num) , 'w')
    else:
        logger = Tee(os.path.join(args.snapshot_dir,'test_epoch_%d.txt' % args.restore_epoch),'w')

    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))

    test(args)