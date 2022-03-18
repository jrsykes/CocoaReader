#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 15:09:31 2022

@author: jamiesykes
"""
import os
from PIL import Image
import time
import shutil

base_dir = '/local/scratch/jrs596/dat/ResNetFung50+_images'

classes = os.listdir(base_dir)

for class_ in classes:
    images = os.listdir(os.path.join(base_dir, class_))
    for image in images:
        with Image.open(os.path.join(base_dir, class_, image)) as im:
            width, height = im.size
            if width > 600 and height > 600:
                source = os.path.join(base_dir, class_, image)
                dest = os.path.join('/local/scratch/jrs596/dat/Forestry_ArableImages_GoogleBing_Licenced/', class_, str(time.time()) + '.jpeg')
                shutil.copy(source, dest)
      