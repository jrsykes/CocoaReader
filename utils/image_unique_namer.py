#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 14:46:31 2022

@author: jamiesykes
"""

import os
import time


root ='/home/jamie/Documents/cacao_net/compiled_cocoa_images/'

dirs = os.listdir('/home/jamie/Documents/cacao_net/compiled_cocoa_images')

count = 1
for i in dirs:
    images = os.listdir(root + i)
    
    for j in images:
        os.rename(root + i + '/' + j, root + '/' + i + '/' + i + str(time.time()) + str(count) + '.jpeg')
        count += 1

