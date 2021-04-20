#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : YoutubeVOS.py
from pycocotools.ytvos import YTVOS
from torch.utils.data import Dataset
import os
import numpy as np
import random
from PIL import Image


class YTVOSDataset(Dataset):
    def __init__(self, data_path=None, train=True, valid=False,
                 set_index=1, finetune_idx=None,
                 support_frame=5, query_frame=1, sample_per_class=10,
                 transforms=None, another_transform=None):
        self.train = train
        self.valid = valid
        self.set_index = set_index
        self.support_frame = support_frame
        self.query_frame = query_frame
        self.sample_per_class = sample_per_class
        self.transforms = transforms
        self.another_transform = another_transform

        if data_path is None:
            data_path = os.path.join(os.path.expanduser('~'), 'Lab/DANet')
        data_dir = os.path.join(data_path, 'data', 'Youtube-VOS')
        self.img_dir = os.path.join(data_dir, 'train', 'JPEGImages')
        self.ann_file = os.path.join(data_dir, 'train', 'train.json')

        self.load_annotations()

        print('data set index: ', set_index)
        self.train_list = [n + 1 for n in range(40) if n % 4 != (set_index - 1)]
        self.valid_list = [n + 1 for n in range(40) if n % 4 == (set_index - 1)]

        if train and not valid:
            self.class_list = self.train_list
        else:
            self.class_list = self.valid_list
        if finetune_idx is not None:
            self.class_list = [self.class_list[finetune_idx]]

        self.video_ids = []
        for class_id in self.class_list:
            tmp_list = self.ytvos.getVidIds(catIds=class_id)
            tmp_list.sort()
            self.video_ids.append(tmp_list)  # list[list[video_id]]
        if not self.train:
            self.test_video_classes = []
            for i in range(len(self.class_list)):
                for j in range(len(self.video_ids[i]) - support_frame):  # remove the support set
                    self.test_video_classes.append(i)

        if self.train:
            self.length = len(self.class_list) * sample_per_class
        else:
            self.length = len(self.test_video_classes)  # test

    def load_annotations(self):
        self.ytvos = YTVOS(self.ann_file)
        self.vid_ids = self.ytvos.getVidIds()  # list[2238] begin : 1
        self.vid_infos = self.ytvos.vids  # vids
        for vid, vid_info in self.vid_infos.items():  # for each vid
            vid_name = vid_info['file_names'][0].split('/')[0]  # '0043f083b5'
            vid_info['dir'] = vid_name
            frame_len = vid_info['length']  # int
            frame_object, frame_class = [], []
            for i in range(frame_len): frame_object.append([])
            for i in range(frame_len): frame_class.append([])
            category_set = set()
            annos = self.ytvos.vidToAnns[vid]  # list[]
            for anno in annos:  # instance_level anns
                assert len(anno['segmentations']) == frame_len, (
                vid_name, len(anno['segmentations']), vid_info['length'])
                for frame_idx in range(frame_len):
                    anno_segmentation = anno['segmentations'][frame_idx]
                    if anno_segmentation is not None:
                        frame_object[frame_idx].append(anno['id'])  # add instance to vid_frame
                        frame_class[frame_idx].append(anno['category_id'])  # add instance class to vid_frame
                        category_set = category_set.union({anno['category_id']})
            vid_info['objects'] = frame_object
            vid_info['classes'] = frame_class
            class_frame_id = dict()
            for class_id in category_set:  # frames index for each class
                class_frame_id[class_id] = [i for i in range(frame_len) if class_id in frame_class[i]]
            vid_info['class_frames'] = class_frame_id

    def get_GT_byclass(self, vid, class_id, frame_num=1, test=False):
        vid_info = self.vid_infos[vid]
        frame_list = vid_info['class_frames'][class_id]
        frame_len = len(frame_list)
        choice_frame = random.sample(frame_list, 1)
        if test:
            frame_num = frame_len
        if frame_num > 1:
            if frame_num <= frame_len:
                choice_idx = frame_list.index(choice_frame[0])
                if choice_idx < frame_num:
                    begin_idx = 0
                    end_idx = frame_num
                else:
                    begin_idx = choice_idx - frame_num + 1
                    end_idx = choice_idx + 1
                choice_frame = [frame_list[n] for n in range(begin_idx, end_idx)]
            else:
                choice_frame = []
                for i in range(frame_num):
                    if i < frame_len:
                        choice_frame.append(frame_list[i])
                    else:
                        choice_frame.append(frame_list[frame_len - 1])
        frames = [np.array(Image.open(os.path.join(self.img_dir, vid_info['file_names'][frame_idx]))) for frame_idx in
                  choice_frame]
        masks = []
        for frame_id in choice_frame:
            object_ids = vid_info['objects'][frame_id]
            mask = None
            for object_id in object_ids:
                ann = self.ytvos.loadAnns(object_id)[0]
                if ann['category_id'] not in self.class_list:
                    continue
                track_id = 1
                if ann['category_id'] != class_id:
                    track_id = 0
                temp_mask = self.ytvos.annToMask(ann, frame_id)
                if mask is None:
                    mask = temp_mask * track_id
                else:
                    mask += temp_mask * track_id

            assert mask is not None
            mask[mask > 0] = 1
            masks.append(mask)

        return frames, masks

    def __gettrainitem__(self, idx):
        list_id = idx // self.sample_per_class
        vid_set = self.video_ids[list_id]

        query_vid = random.sample(vid_set, 1)
        support_vid = random.sample(vid_set, self.support_frame)

        query_frames, query_masks = self.get_GT_byclass(query_vid[0], self.class_list[list_id], self.query_frame)

        support_frames, support_masks = [], []
        for i in range(self.support_frame):
            one_frame, one_mask = self.get_GT_byclass(support_vid[i], self.class_list[list_id], 1)
            support_frames += one_frame
            support_masks += one_mask

        if self.transforms is not None:
            query_frames, query_masks = self.transforms(query_frames, query_masks)
            support_frames, support_masks = self.transforms(support_frames, support_masks)
        return query_frames, query_masks, support_frames, support_masks, self.class_list[list_id]

    def __gettestitem__(self, idx):
        # random.seed()
        begin_new = False
        if idx == 0:
            begin_new = True
        else:
            if self.test_video_classes[idx] != self.test_video_classes[idx - 1]:
                begin_new = True
        list_id = self.test_video_classes[idx]
        vid_set = self.video_ids[list_id]

        support_frames, support_masks = [], []
        if begin_new:
            support_vid = random.sample(vid_set, self.support_frame)
            query_vids = []
            for id in vid_set:
                if not id in support_vid:
                    query_vids.append(id)
            self.query_ids = query_vids
            self.query_idx = -1
            for i in range(self.support_frame):
                one_frame, one_mask = self.get_GT_byclass(support_vid[i], self.class_list[list_id], 1)
                support_frames += one_frame
                support_masks += one_mask

        self.query_idx += 1
        query_vid = self.query_ids[self.query_idx]
        query_frames, query_masks = self.get_GT_byclass(query_vid, self.class_list[list_id], test=True)

        if self.transforms is not None:
            query_frames, query_masks = self.transforms(query_frames, query_masks)
            if begin_new:
                if self.another_transform is not None:
                    support_frames, support_masks = self.another_transform(support_frames, support_masks)
                else:
                    support_frames, support_masks = self.transforms(support_frames, support_masks)
        vid_info = self.vid_infos[query_vid]
        vid_name = vid_info['dir']
        return query_frames, query_masks, support_frames, support_masks, self.class_list[list_id], vid_name, begin_new

    def __getitem__(self, idx):
        if self.train:
            return self.__gettrainitem__(idx)
        else:
            return self.__gettestitem__(idx)

    def __len__(self):
        return self.length

    def get_class_list(self):
        return self.class_list


if __name__ == "__main__":
    ytvos = YTVOSDataset(train=True, query_frame=5, support_frame=5)
    for i in range(len(ytvos)):
        video_query_img, video_query_mask, new_support_img, new_support_mask, idx, vid, begin_new = ytvos[i]