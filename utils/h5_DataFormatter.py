#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:03:57 2021

@author: jamie
"""
import cv2
import datetime as dt
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
import os
import pandas as pd
from glob import glob
import random
from PIL import Image
#%%

images = []
path = '/local/scratch/jrs596/dat/ResNetFung50+_images/'
classes = os.listdir(path)

for i in classes:
    dir_ = path + i
    for j in os.listdir(dir_):
        
        im = Image.open(dir_ + '/' + j)
        if len(im.getpixel((1, 1))) == 3:       
            images.append(dir_ + '/' + j)
        else:
            print('Not .jpeg: ' + j)
            print(im.getpixel((1, 1)))





random.shuffle(images)

n=len(images)
train = images[:int(n*0.8)]
val = images[int(n*0.8):int(n*0.9)]
test = images[int(n*0.9):]

#%%

def h5DataFormatter(filename, images_, division):
    try:
        os.remove(filename)
    except:
        pass

    HEIGHT = 300
    WIDTH = 300
    SHAPE = (HEIGHT, WIDTH, 3)
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset(
                    name='list_classes',
                    data=classes)
        

        image_list = []
        for i, im in enumerate(images_):
            image = Image.open(im)
 
            image = np.array(image.resize((HEIGHT, WIDTH)))
            image_list.append(image)
                        
        hf.create_dataset(
                    name= division + '_set_x',
                    data=image_list,
                    compression="gzip",
                    compression_opts=9) 
        
        image_list = []

        y_list = []
        for i,img in enumerate(images_):
            if 'AauberginesDiseased' in img:
 	            y =1
            if 'StrawberriesDiseased' in img:
                y=2
            if 'GrapesDiseased' in img:
                y=3
            if 'CherryHealthy' in img:
                y=4
            if 'PeasHealthy' in img:
                y=5
            if 'CocoaHealthy' in img:
                y=6
            if 'RapeseedHealthy' in img:
	            y=7
            if 'SunflowerDiseased' in img:
	            y=8
            if 'BarleyDiseased' in img:
             	y=9
            if 'CucumbersDiseased' in img:
                y=10
            if 'OnionsDiseased' in img:
             	y =11
            if 'ApplesDiseased' in img:
	            y =12
            if 'PeachesHealthy' in img:
                y=13
            if 'PeachesDiseased' in img:
                y=14
            if 'MangosHealthy' in img:
                y=15
            if 'OnionsHealthy' in img:
                y=16
            if 'AauberginesHealthy' in img:
                y=17
            if 'TobaccoDiseased' in img:
                y=18
            if 'BananasHealthy' in img:
                y=19
            if 'Sweet_potatoesHealthy' in img:
                y=20
            if 'WatermelonsDiseased' in img:
                y=21
            if 'WatermelonsHealthy' in img:
                y=22
            if 'PotatoesDiseased' in img:
                y=23
            if 'MangosDiseased' in img:
                y=24
            if 'TobaccoHealthy' in img:
                y=25
            if 'SoybeansDiseased' in img:
                y=26
            if 'TomatoesDiseased' in img:
                y=27
            if 'MaizeHealthy' in img:
                y=28
            if 'CassavaHealthy' in img:
                y=29
            if 'OlivesHealthy' in img:
                y=30
            if 'WheatHealthy' in img:
                y=31
            if 'CherryDiseased' in img:
                y=32
            if 'CabbagesHealthy' in img:
                y=33
            if 'OrangesHealthy' in img:
                y=34
            if 'GrapesHealthy' in img:
                y=35
            if 'OrangesDiseased' in img:
                y=36
            if 'StrawberriesHealthy' in img:
                y=37
            if 'MaizeDiseased' in img:
                y=38
            if 'PotatoesHealthy' in img:
                y=39
            if 'RapeseedDiseased' in img:
                y=40
            if 'SugarcaneDiseased' in img:
                y=41
            if 'WheatDiseased' in img:
                y=42
            if 'PeasDiseased' in img:
                y=43
            if 'RiceDiseased' in img:
                y=44
            if 'ApplesHealthy' in img:
                y=45
            if 'SpinachDiseased' in img:
                y=46
            if 'SunflowerHealthy' in img:
                y=47
            if 'LettuceHealthy' in img:
                y=48
            if 'SugarcaneHealthy' in img:
                y=49
            if 'RiceHealthy' in img:
                y=50
            if 'OlivesDiseased' in img:
                y=51
            if 'CassavaDiseased' in img:
                y=52
            if 'SpinachHealthy' in img:
                y=53
            if 'BananasDiseased' in img:
                y=54
            if 'TeaDiseased' in img:
                y=55
            if 'CocoaDiseased' in img:
                y=56
            if 'CabbagesDiseased' in img:
                y=57
            if 'BarleyHealthy' in img:
                y=58
            if 'TeaHealthy' in img:
                y=59
            if 'LettuceDiseased' in img:
                y=60
            if 'TomatoesHealthy' in img:
                y=61
            if 'Sweet_potatoesDiseased' in img:
                y=62
            if 'GarlicDiseased' in img:
                y=63
            if 'SoybeansHealthy' in img:
                y=64
            if 'GarlicHealthy' in img:
                y=65
            if 'CucumbersHealthy' in img:
                y=66
            y_list.append(y.astype(int))
        
        
        hf.create_dataset(
                    name= division + '_set_y',
                    data=y_list)


h5DataFormatter('/local/scratch/jrs596/dat/train.h5', train, 'train')
h5DataFormatter('/local/scratch/jrs596/dat/val.h5', val, 'val')
h5DataFormatter('/local/scratch/jrs596/dat/test.h5', test, 'test')

#%%
#def load_dataset():
#    train_dataset = h5py.File('train.h5', "r")
#    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
#    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels#

#    test_dataset = h5py.File('test.h5', "r")
#    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
#    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels#

#    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
#    
#    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
#    
#    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes#

#def convert_to_one_hot(Y, C):
#    Y = np.eye(C)[Y.reshape(-1)].T
#    return Y#
#

##%%#

#X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()#
#
#

## Normalize image vectors
## X_train = X_train_orig/255.
## X_test = X_test_orig/255.#

## Convert training and test labels to one hot matrices
#Y_train = convert_to_one_hot(Y_train_orig, len(classes)).T
#Y_test = convert_to_one_hot(Y_test_orig, 67).T#

#print(Y_test_orig.shape)#

#print ("number of training examples = " + str(X_train.shape[0]))
#print ("number of test examples = " + str(X_test.shape[0]))
#print ("X_train shape: " + str(X_train.shape))
#print ("Y_train shape: " + str(Y_train.shape))
#print ("X_test shape: " + str(X_test.shape))
#print ("Y_test shape: " + str(Y_test.shape))#

##%%#

#a = np.eye(66)
## print(a)
#Y=Y_test_orig#

#print(Y[:,5000])



