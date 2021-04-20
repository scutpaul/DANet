#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : DAN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import libs.models.DAN.resnet as models
from libs.models.DAN.decoder import Decoder

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        resnet = models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2,
                                    resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)

        self.layer1 = resnet.layer1  # 1/4, 256
        self.layer2 = resnet.layer2  # 1/8, 512
        self.layer3 = resnet.layer3  # 1/16, 1024 --> 1/8
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    def forward(self, in_f):

        f = in_f
        x = self.layer0(f)
        l1 = self.layer1(x)  # 1/4, 256
        l2 = self.layer2(l1)  # 1/8, 512
        l3 = self.layer3(l2)  # 1/8, 1024

        return l3, l3, l2, l1

class QueryKeyValue(nn.Module):
    # Not using location
    def __init__(self, indim, keydim, valdim):
        super(QueryKeyValue, self).__init__()
        self.query = nn.Conv2d(indim, keydim,kernel_size=3, padding=1,stride=1)
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.query(x),self.Key(x), self.Value(x)

class DAN(nn.Module):
    def __init__(self):
        super(DAN, self).__init__()
        self.encoder = Encoder() # output 2048
        encoder_dim = 1024
        h_encdim = int(encoder_dim/2)
        self.support_qkv = QueryKeyValue(encoder_dim,keydim=128,valdim=h_encdim)
        self.query_qkv = QueryKeyValue(encoder_dim,keydim=128,valdim=h_encdim)

        self.conv_q = nn.Conv2d(encoder_dim,h_encdim,kernel_size=1, stride=1,padding=0)

        # low_level_features to 48 channels
        self.Decoder = Decoder(encoder_dim, 256)

    def transformer(self, Q, K, V):
        # Q : B CQ WQ
        # K : B WK CQ
        # V : B CV WK
        B, CQ, WQ = Q.shape
        _, CV, WK = V.shape

        P = torch.bmm(K, Q)  # B WK WQ
        P = P / math.sqrt(CQ)
        P = torch.softmax(P, dim=1)

        M = torch.bmm(V, P)  # B CV WQ

        return M, P

    def forward(self, img, support_image, support_mask,time=None):
        batch, frame, in_channels, height, width = img.shape
        _, sframe, mask_channels, Sheight,Swidth = support_mask.shape
        assert  height == Sheight
        assert width == Swidth
        batch_frame = batch*frame
        img = img.view(-1, in_channels, height, width)
        support_image = support_image.view(-1, in_channels, height, width)
        support_mask = support_mask.view(-1, mask_channels, height, width)
        # img flow : [batch, frame, channels, height, width] -> [batch*frame, channels, height, width]
        # support_mask : [batch, sframe, mask_channels, height, width]
        in_f = torch.cat((img,support_image),dim=0)
        encoder_f,encoder_f_l3,encoder_f_l2,encoder_f_l1 = self.encoder(in_f)
        if time is not None:
            time.t1()
        support_mask = F.interpolate(support_mask, encoder_f.size()[2:], mode='bilinear', align_corners=True)
        query_feat_l1 = encoder_f_l1[:batch_frame]
        query_feat_l2 = encoder_f_l2[:batch_frame]
        query_feat = encoder_f[:batch_frame]
        support_feat = encoder_f[batch_frame:]


        support_fg_feat = support_feat * support_mask
        support_bg_feat = support_feat * (1-support_mask)

        # q,k = batch*frames, 128, h/16, w/16
        # v = batch*frames, 256, h/16, w/16
        _, support_k, support_v = self.support_qkv(support_fg_feat)
        query_q, query_k, query_v = self.query_qkv(query_feat)
        _,_,qh,qw = query_k.shape
        _, c, h,w = support_k.shape
        _, vc, _, _ = support_v.shape

        assert qh == h and qw == w
        # transforms query_middle_q to support_kv
        # support [b*f c h w] -> [b f c h w] -> [b c f h w] -> [b c WF]
        support_k = support_k.view(batch, sframe, c, h, w)
        support_v = support_v.view(batch, sframe, vc, h, w)
        # B, WK, CK
        support_k = support_k.permute(0, 2, 1, 3, 4).contiguous().view(batch, c, -1).permute(0, 2, 1).contiguous()
        # B, CV, WK
        support_v = support_v.permute(0, 2, 1, 3, 4).contiguous().view(batch, vc, -1)
        middle_frame_index = int(frame/2)
        query_q = query_q.view(batch, frame, c, h, w)
        query_k = query_k.view(batch,frame,c, h, w)
        middle_q = query_q[:,middle_frame_index]
        assert len(middle_q.shape) == 4
        # B, CQ, WQ
        middle_q = middle_q.view(batch, c, -1)
        # B CV WQ --> V
        new_V, sim_refer = self.transformer(middle_q,support_k,support_v)
        # print(sim_refer.shape)
        # transform query_qkv to query_middle_kv
        # B WK CK
        middle_K = query_k[:,middle_frame_index]
        middle_K = middle_K.view(batch, c, -1).permute(0, 2, 1).contiguous()

        query_q = query_q.permute(0,2,1,3,4).contiguous().view(batch,c,-1)
        Out,sim_middle = self.transformer(query_q, middle_K, new_V)
        after_transform = Out.view(batch,vc,frame,h,w)
        after_transform = after_transform.permute(0, 2, 1, 3, 4).contiguous()

        # [batch*frames, 1024, h/16,w/16]
        query_feat = self.conv_q(query_feat)
        after_transform = after_transform.view(-1, vc, h, w)
        after_transform = torch.cat((after_transform, query_feat),dim=1)
        # aspp
        # x = self.aspp(after_transform)
        if time is not None:
            time.t2()
        x = self.Decoder(after_transform, query_feat_l2, query_feat_l1, img)

        pred_map = torch.nn.Sigmoid()(x)

        # batch, frame, outchannel, height, width
        pred_map = pred_map.view(batch, frame, 1, height, width)
        return pred_map

if __name__ == '__main__':
    model = DAN()
    img = torch.FloatTensor(2,3,3,224,224)
    support_mask = torch.FloatTensor(2,5,1,30,30)
    support_img = torch.FloatTensor(2,5,3,30,30)
    pred_map= model(img, support_img,support_mask)
    print(pred_map.shape)