#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:44:49 2021

@author: jamie
"""

import os
from shutil import copyfile

base_dir = '/home/jamie/Documents/cacao_net/Cocoa Ripeness Dataset/'
pics = os.listdir(base_dir)

S = "/home/jamie/Documents/cacao_net/pod_ripness_dataset_org/S/"
M = "/home/jamie/Documents/cacao_net/pod_ripness_dataset_org/M/"
I = "/home/jamie/Documents/cacao_net/pod_ripness_dataset_org/I/"

for i in pics:
    if 'I' in i:
        copyfile(base_dir + i, I + i)
    elif 'M' in i:
        copyfile(base_dir + i, M + i)
    elif 'S' in i:
        copyfile(base_dir + i, S + i)
        
