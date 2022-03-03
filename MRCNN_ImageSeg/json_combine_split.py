#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:57:53 2022

@author: jamiesykes
"""

import json
import glob
import os
import shutil
from random import shuffle
import time

result = []

for f in glob.glob("/local/scratch/jrs596/MaskRCNN/dat/cvat_out/*/annotations/*json"):
    with open(f, "rb") as infile:
        result.append(json.load(infile))


dict1 = result[1]



#%% Reassign ids for all images

id_ = 1
images_list = [] 
annotations_list = []   
for i in result:
    a = i['images']
    b = i['annotations']
    for j in a:
        old_image_id = j['id']
        j['id'] = time.time() + id_
        for k in b:
            if k['image_id'] == old_image_id:
                k['image_id'] = j['id']
        id_ += 1
    
    images_list.append(a)
    annotations_list.append(b)
    
images_list = [item for sublist in images_list for item in sublist]
annotations_list = [item for sublist in annotations_list for item in sublist]

count =1
for i in annotations_list:
    i['id'] = count
    count += 1




#%% Create combined dictionary

key_list = []
for key, value in dict1.items():
    key_list.append(key)

new_dict = { key_list[i] : '' for i in range(0, len(key_list) ) }
new_dict['licenses'] = dict1['licenses']
new_dict['info'] = dict1['info']
new_dict['categories'] = dict1['categories']
new_dict['images'] = images_list
new_dict['annotations'] = annotations_list  

n_images = len(images_list)


#%% Create split dictionaries

image_names = []
for i in new_dict['images']:
    image_names.append(i['id'])


shuffle(image_names)

train_ids = image_names[:int(len(image_names)*0.8)]
val_ids = image_names[int(len(image_names)*0.8):int(len(image_names)*0.9)]
test_ids = image_names[int(len(image_names)*0.9):]


#%%
image_list = []
for i in new_dict['images']:
    if i['id'] in train_ids:
        image_list.append(i)
        
train_dict = {}
train_dict['licenses'] = dict1['licenses']
train_dict['info'] = dict1['info']
train_dict['categories'] = dict1['categories']
train_dict['images'] = image_list

image_list = []
for i in new_dict['images']:
    if i['id'] in val_ids:
        image_list.append(i)
val_dict = {}
val_dict['licenses'] = dict1['licenses']
val_dict['info'] = dict1['info']
val_dict['categories'] = dict1['categories']
val_dict['images'] = image_list

image_list = []
for i in new_dict['images']:
    if i['id'] in test_ids:
        image_list.append(i)
test_dict = {}
test_dict['licenses'] = dict1['licenses']
test_dict['info'] = dict1['info']
test_dict['categories'] = dict1['categories']
test_dict['images'] = image_list

#%% populate split dictionaries with annotations

train_list = []
val_list = []
test_list = []

for i in  new_dict['annotations']:
    if i['image_id'] in train_ids:
        train_list.append(i)
    if i['image_id'] in val_ids:
        val_list.append(i)
    if i['image_id'] in test_ids:
        test_list.append(i)

train_dict['annotations'] = train_list
val_dict['annotations'] = val_list
test_dict['annotations'] = test_list


#%% Testing

# for i in val_dict['images']:
#     print('Image ID: ' + str(i['id']))
#     print('Filename: ' + i['file_name'])
    
#     print('\n')

# for i in val_dict['annotations']:
#     print('Image ID: ' + str(i['image_id']))
#     print('Annotation ID: ' + str(i['id']))

#     print('\n')
    # print(i)
    # print('\n')
#%% Save new annotation files

with open('/local/scratch/jrs596/MaskRCNN/dat/results/annotations/train_combined_instances_default.json', 'w') as json_file:
  json.dump(train_dict, json_file)
  
with open('/local/scratch/jrs596/MaskRCNN/dat/results/annotations/val_combined_instances_default.json', 'w') as json_file:
  json.dump(val_dict, json_file)
  
with open('/local/scratch/jrs596/MaskRCNN/dat/results/annotations/test_combined_instances_default.json', 'w') as json_file:
  json.dump(test_dict, json_file)
  
  
#%% Split images to match annotation files

root_dir = '/local/scratch/jrs596/MaskRCNN/dat/cvat_out/'

# Get all image files
image_files = {}
for root, subdirs, files in os.walk(root_dir):
    for i in files:
        if '.jpeg' in i:
            image_files[i] = root + '/' + i
            
# Train
train_images = []
for i in train_dict['images']:
    train_images.append([i['file_name']])
train_images = [item for sublist in train_images for item in sublist]

for i in train_images:
    shutil.copyfile(image_files[i], '/local/scratch/jrs596/MaskRCNN/dat/results/images/train/' + i)


# Val

val_images = []
for i in val_dict['images']:
    val_images.append([i['file_name']])
val_images = [item for sublist in val_images for item in sublist]

for i in val_images:
    shutil.copyfile(image_files[i], '/local/scratch/jrs596/MaskRCNN/dat/results/images/val/' + i) 
  
  
# Test 
  
test_images = []
for i in test_dict['images']:
    test_images.append([i['file_name']])
test_images = [item for sublist in test_images for item in sublist]

for i in test_images:
    shutil.copyfile(image_files[i], '/local/scratch/jrs596/MaskRCNN/dat/results/images/test/' + i)
  
  
  
  
  