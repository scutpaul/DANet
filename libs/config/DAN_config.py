#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : DAN_config.py
import os


from easydict import EasyDict
OPTION = EasyDict()
OPTION.root_path = os.path.join(os.path.expanduser('~'), 'Lab/DANet')
OPTION.input_size = (241,425)
OPTION.test_size = (241,425)
OPTION.SNAPSHOT_DIR = os.path.join(OPTION.root_path, 'DANsnapshots')
