#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : transform.py
import numpy as np
import torch
import cv2

import random
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class Compose(object):
    """
    Combine several transformation in a serial manner
    """

    def __init__(self, transform=[]):
        self.transforms = transform

    def __call__(self, imgs, annos):

        for m in self.transforms:
            imgs, annos = m(imgs, annos)

        return imgs, annos

class Transpose(object):

    """
    transpose the image and mask
    """

    def __call__(self, imgs, annos):

        H, W, _ = imgs[0].shape
        if H < W:
            return imgs, annos
        else:
            timgs = [np.transpose(img, [1, 0, 2]) for img in imgs]
            tannos = [np.transpose(anno, [1, 0, 2]) for anno in annos]

            return timgs, tannos

class RandomAffine(object):

    """
    Affine Transformation to each frame
    """

    def __call__(self, imgs, annos):


        seq = iaa.Sequential([
            iaa.Crop(percent=(0.0, 0.1), keep_size=True),
            iaa.Affine(scale=(0.95, 1.05), shear=(-10, 10), rotate=(-15, 15))
        ])

        seq = seq.to_deterministic()

        num = len(imgs)
        for idx in range(1, num):
            img = imgs[idx]
            anno = annos[idx]


            segmap = SegmentationMapsOnImage(anno, shape=img.shape)
            img_aug, segmap_aug = seq(image=img, segmentation_maps=segmap)
            imgs[idx] = img_aug
            annos[idx] = segmap_aug.get_arr()

        return imgs, annos

class AdditiveNoise(object):
    """
    sum additive noise
    """

    def __init__(self, delta=5.0):
        self.delta = delta
        assert delta > 0.0

    def __call__(self, imgs, annos):
        v = np.random.uniform(-self.delta, self.delta)
        for id, img in enumerate(imgs):
            imgs[id] += v

        return imgs, annos


class RandomContrast(object):
    """
    randomly modify the contrast of each frame
    """

    def __init__(self, lower=0.97, upper=1.03):
        self.lower = lower
        self.upper = upper
        assert self.lower <= self.upper
        assert self.lower > 0

    def __call__(self, imgs, annos):
        v = np.random.uniform(self.lower, self.upper)
        for id, img in enumerate(imgs):
            imgs[id] *= v

        return imgs, annos


class RandomMirror(object):
    """
    Randomly horizontally flip the video volume
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos):

        v = random.randint(0, 1)
        if v == 0:
            return imgs, annos

        sample = imgs[0]
        h, w = sample.shape[:2]

        for id, img in enumerate(imgs):
            imgs[id] = img[:, ::-1, :]

        for id, anno in enumerate(annos):
            annos[id] = anno[:, ::-1, :]

        return imgs, annos

class ToFloat(object):
    """
    convert value type to float
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos):
        for idx, img in enumerate(imgs):
            imgs[idx] = img.astype(dtype=np.float32, copy=True)

        for idx, anno in enumerate(annos):
            annos[idx] = anno.astype(dtype=np.float32, copy=True)

        return imgs, annos

class Rescale(object):

    """
    rescale the size of image and masks
    """

    def __init__(self, target_size):
        assert isinstance(target_size, (int, tuple, list))
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, imgs, annos):

        h, w = imgs[0].shape[:2]
        new_height, new_width = self.target_size

        factor = min(new_height / h, new_width / w)
        height, width = int(factor * h), int(factor * w)
        pad_l = (new_width - width) // 2
        pad_t = (new_height - height) // 2

        for id, img in enumerate(imgs):
            canvas = np.zeros((new_height, new_width, 3), dtype=np.float32)
            rescaled_img = cv2.resize(img, (width, height))
            canvas[pad_t:pad_t+height, pad_l:pad_l+width, :] = rescaled_img
            imgs[id] = canvas

        for id, anno in enumerate(annos):
            canvas = np.zeros((new_height, new_width, 1), dtype=np.float32)
            rescaled_anno = cv2.resize(anno, (width, height), cv2.INTER_NEAREST)
            canvas[pad_t:pad_t + height, pad_l:pad_l + width, :] = rescaled_anno[:,:,np.newaxis]
            annos[id] = canvas

        return imgs, annos


class Stack(object):

    """
    stack adjacent frames into input tensors
    """

    def __call__(self, imgs, annos):

        num_img = len(imgs)
        num_anno = len(annos)

        h, w, = imgs[0].shape[:2]

        assert num_img == num_anno
        img_stack = np.stack(imgs, axis=0)
        anno_stack = np.stack(annos, axis=0)

        return img_stack, anno_stack

class ToTensor(object):

    """
    convert to torch.Tensor
    """

    def __call__(self, imgs, annos):

        imgs = torch.from_numpy(imgs.copy())
        annos = torch.from_numpy(annos.astype(np.uint8, copy=True)).float()

        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        annos = annos.permute(0, 3, 1, 2).contiguous()

        return imgs, annos

class Normalize(object):

    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3]).astype(np.float32)
        self.std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3]).astype(np.float32)

    def __call__(self, imgs, annos):

        for id, img in enumerate(imgs):
            imgs[id] = (img / 255.0 - self.mean) / self.std

        return imgs, annos

class ReverseClip(object):

    def __call__(self, imgs, annos):

        return imgs[::-1], annos[::-1]



class AddAxis(object):

    def __call__(self, imgs, annos):
        for idx, anno in enumerate(annos):
            annos[idx] = anno[:,:,np.newaxis]
        return imgs, annos

class TrainTransform(object):

    def __init__(self, size):
        self.transform = Compose([
            AddAxis(),
            Transpose(),
            RandomAffine(),
            ToFloat(),
            RandomContrast(),
            AdditiveNoise(),
            RandomMirror(),
            Rescale(size),
            Normalize(),
            Stack(),
            ToTensor(),
        ])

    def __call__(self, imgs, annos):
        return self.transform(imgs, annos)


class TestTransform(object):

    def __init__(self, size):
        self.transform = Compose([
            AddAxis(),
            ToFloat(),
            Rescale(size),
            Normalize(),
            Stack(),
            ToTensor(),
        ])

    def __call__(self, imgs, annos):
        return self.transform(imgs, annos)

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class ReverseToImage(object):
    def __init__(self):
        self.transform = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self,imgs):
        return self.transform(imgs)