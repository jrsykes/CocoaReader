#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 19:50:06 2020

@author: jamie
"""

import os
import random
import shutil

split = .8
base_dir = '/home/jamie/Documents/weed_net/weed_net_data/'
species = os.listdir(base_dir)

for item in species:
    os.makedirs(base_dir + '/train/' + item)
    os.makedirs(base_dir + '/test/' + item)
    
    files = os.listdir(base_dir + item)
    #files.remove('test')
    #files.remove('train')
    random.shuffle(files)
    n_pics = len(files)
    train = files[:int(n_pics*split)]
    test = files[int(n_pics*split):]
        
    for i in train:
        shutil.move(base_dir + item + '/' + i, base_dir + '/train/' + item)
    for i in test:
        shutil.move(base_dir + item + '/' + i, base_dir + '/test/' + item)
    os.rmdir(base_dir + item)
