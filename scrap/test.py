#%%
import torch
from torchvision import datasets, models, transforms
import os
from torch import nn
from sklearn import metrics
import pandas as pd
import time
from matplotlib import pyplot as plt 
import time
import random
import shutil
root = '/local/scratch/jrs596/dat/tmp/norm'
data_dir = "/local/scratch/jrs596/dat/FAIGB_FinalSplit"


input_size = 20
batch_size = 50

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ]),
}
#%%
# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
	for x in ['train', 'val']}
# Create training and validation dataloaders

#count number of diseased and healthy images in image_datasets['train']

n_classes = 4

def subset_classes(n_classes, train_dataset):
    classes = image_datasets['train'].classes
    classes_tensor = torch.tensor(list(range(len(classes[0:n_classes]))))
    indices = (torch.tensor(image_datasets['train'].targets)[..., None] == classes_tensor).any(-1).nonzero(as_tuple=True)[0]
    train_subset = torch.utils.data.Subset(image_datasets['train'], indices)
    return train_subset

train_subset = subset_classes(n_classes, image_datasets['train'])

for img, label in train_subset:
    print(label)
    print(img.shape)
    break


#%%
class_dict = {}
for class_ in image_datasets['train'].classes:
    class_dict[class_] = 0

for i in image_datasets['train'].imgs:
    if i[1] == 1:
        healthy += 1
    else:
        diseased += 1

#print list of images in image_datasets['train']
diseased, healthy = 0, 0
original = image_datasets['train'].imgs
random.shuffle(original)
subset =[]
for i in original:
    if i[1] == 1:
        if healthy < 4443:
            healthy += 1
            subset.append(i)
    else:
        diseased += 1
        subset.append(i)

image_datasets['train'].imgs = subset

print(len(subset))
#%%

train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4)
print(len(train_loader.dataset.imgs))


for inputs, labels in train_loader:
    print(labels)
    #print percentagle of diseased and healthy images in batch
    diseased, healthy = 0, 0
    for label in labels:
        if label == 1:
            healthy += 1
        else:
            diseased += 1
    print(diseased/(diseased+healthy)*100, healthy/(diseased+healthy)*100)
    
    break
# %%

n_diseased = len(os.listdir(data_dir + '/train/Diseased'))

healthy_imgs = os.listdir(data_dir + '/FullTrainHealthy')
random.shuffle(healthy_imgs)

shutil.rmtree(data_dir + '/train/Healthy')
os.mkdir(data_dir + '/train/Healthy')

for i in healthy_imgs[:n_diseased]:
    shutil.copy(os.path.join(data_dir, 'FullTrainHealthy', i), os.path.join(data_dir, 'train/Healthy/', i))



# %%
# print number of CPUs#
import os
print(os.cpu_count()-4)
# %%
