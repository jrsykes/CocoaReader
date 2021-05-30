#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 20:34:59 2020

@author: jamie
"""
import os

base_dir = '/home/jamie/Documents/cacao_net'
species = os.listdir(base_dir)

from PIL import Image
import io

for i in species:
    files = os.listdir(base_dir + i)
    for item in files:
        
        # if '.png' in item:b
        #     os.remove(base_dir + i + '/' + item)
        #     print('Removed file: ' + i + '/' + item)
        # if '.gif' in item:
        #     os.remove(base_dir + i + '/' + item)
        #     print('Removed file: ' + i + '/' + item)
        
        print(i + '/' + item)
        byteImgIO = io.BytesIO()
        byteImg = Image.open(base_dir + i + '/' + item)
        byteImg.save(byteImgIO, "JPEG")
        byteImgIO.seek(0)
        byteImg = byteImgIO.read()

        bad_file_list = []
        # Non test code
        dataBytesIO = io.BytesIO(byteImg)
        Image.open(dataBytesIO)

            
#image = Image.open('/home/jamie/Documents/weed_net/weed_net_data/test/aethusa_cynapium/10110.jpg')

print(image.format)
            
 
    
 

            
        