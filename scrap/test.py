import torch
from torchvision import datasets, models, transforms
import os
from torch import nn
from sklearn import metrics


model_path = '/local/scratch/jrs596/ResNetFung50_Torch/models/model.pth'
data_dir = "/local/scratch/jrs596/dat/ResNetFung50+_images_organised_subset"

model = torch.load(model_path)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


input_size = 224
batch_size = 42
criterion = nn.CrossEntropyLoss()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}#

running_loss = 0.0
running_corrects = 0

running_precision = 0
running_recall = 0
running_f1 = 0

phase = 'val'



for inputs, labels in dataloaders_dict[phase]:
	inputs = inputs.to(device)
	labels = labels.to(device)
	outputs = model(inputs)
	loss = criterion(outputs, labels)
	_, preds = torch.max(outputs, 1)#

	running_loss += loss.item() * inputs.size(0)
	running_corrects += torch.sum(preds == labels.data)

	stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
	stats_out = stats['weighted avg']
	running_precision += stats_out['precision']
	running_recall += stats_out['recall']
	running_f1 += stats_out['f1-score']

n = len(dataloaders_dict[phase].dataset)
epoch_loss = float(running_loss / n)
epoch_acc = float(running_corrects.double() / n)
precision = (running_precision*batch_size) / n         
recall = (running_recall*batch_size) / n        
f1 = (running_f1*batch_size) / n

print(epoch_loss)
print(epoch_acc)
print(precision)
print(recall)
print(f1)

