import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import pickle
import numpy as np
from sklearn import metrics
import os
import copy
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import pandas as pd

input_size = 1000
batch_size = 5

root = '/local/scratch/jrs596/dat'
data_dir = os.path.join(root, 'FAIGB_reduced_split/val')
model_path = os.path.join(root, 'models/DisNet_1kdim_HighRes_ConvNext_reduced.pkl')

# Number of classes in the dataset
num_classes = len(os.listdir(data_dir))

#Load model
device = torch.device("cuda")

model = models.convnext_tiny(weights=None)
in_feat = model.classifier[2].in_features
model.classifier[2] = torch.nn.Linear(in_feat, num_classes)


pretrained_model_wts = pickle.load(open(os.path.join(model_path), "rb"))
unpickled_model_wts = copy.deepcopy(pretrained_model_wts['model'])

###############
#Remove 'module.' from layer names
new_keys = []
for key, value in unpickled_model_wts.items():
    new_keys.append(key.replace('module.', ''))

for i in new_keys:
    unpickled_model_wts[i] = unpickled_model_wts.pop('module.' + i)#    
##############
model.load_state_dict(unpickled_model_wts)
print('Checkpint weights loaded')

#Run model on all GPUs
model = nn.DataParallel(model)
model = model.to(device)

model.eval()

params_to_update = model.parameters()
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.ToTensor(),
    ]),
}

# Create training and validation datasets
image_dataset = datasets.ImageFolder(os.path.join(data_dir), data_transforms['val'])
# Create training and validation dataloaders
dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, 
    shuffle=False, num_workers=2)

y_pred = []
y_true = []

for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)  #
    outputs = model(inputs)
    #_, preds = torch.max(outputs, 1)   
    preds = torch.argmax(outputs, dim=1)#
    y_true.append(labels.data.tolist())
    y_pred.append(preds.tolist())
    

y_true = [x for xs in y_true for x in xs]#
y_pred = [x for xs in y_pred for x in xs]##

stats = metrics.classification_report(y_true, y_pred, digits=4, output_dict = True, zero_division = 0)
stats_out = stats['weighted avg']

f1 = []
precision = []
recall = []

for key, value in stats.items():
    try:
        f1.append(value['f1-score'])
        precision.append(value['precision'])
        recall.append(value['recall'])
    except:
        pass


cm = confusion_matrix(y_true, y_pred)#

accs = cm.diagonal()/cm.sum(axis=1)

#print(cm)

names = sorted(os.listdir(data_dir))
#names = ['a', 'b', 'c', 'd', 'e', 'f']

acc_df = pd.DataFrame(columns=['class', 'acc'])
acc_df['class'] = names
acc_df['acc'] = accs
acc_df['f1'] = f1[:-2]
acc_df['precision'] = precision[:-2]
acc_df['recall'] = recall[:-2]

print(acc_df)

df = pd.DataFrame(cm,columns=names)

df.insert(loc=0,column='/',value=names)
print(df)

df.to_csv(os.path.join(root, 'confusion_matrix2.csv'))
acc_df.to_csv(os.path.join(root, 'stats2.csv'))