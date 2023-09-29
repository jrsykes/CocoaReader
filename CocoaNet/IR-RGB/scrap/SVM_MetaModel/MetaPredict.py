#%%
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from joblib import load
import sys
sys.path.append('/users/jrs596/scripts/CocoaReader/utils')
from ArchitectureZoo import DisNet_picoIR
import toolbox

from torchvision import datasets, transforms
import os
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda:0")
#%%

class EnsembleClassifier:
    def __init__(self, efficientnet_b0_path, DisNet_pico, svm_path):
        self.efficientnet_b0 = efficientnet_b0_path
        self.DisNet_pico = DisNet_pico
        self.svm = load(svm_path)
        
    def predict(self, input_tensor):
        # Load and preprocess the image
        # input_batch = input_tensor.unsqueeze(0)
        
        # Move the input to GPU, if available
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Make predictions with efficientnet_b0
        self.efficientnet_b0.eval()
        with torch.no_grad():
            #resize tensor to 424x424
            input_resize = torch.nn.functional.interpolate(input_tensor, size=(424, 424), mode='bilinear', align_corners=False)
            efficientnet_b0_pred = self.efficientnet_b0(input_resize.to(device))
        efficientnet_b0_pred = efficientnet_b0_pred.cpu()
        efficientnet_b0_pred = torch.argmax(efficientnet_b0_pred, dim=1)
        
        # Make predictions with DisNet_pico
        self.DisNet_pico.eval()
        with torch.no_grad():
            DisNet_pico_pred = self.DisNet_pico(input_tensor.to(device))
        DisNet_pico_pred = DisNet_pico_pred.cpu()
        DisNet_pico_pred = torch.argmax(DisNet_pico_pred, dim=1)
        
        # Stack predictions to form feature vector for the SVM
        X_test = np.hstack((efficientnet_b0_pred, DisNet_pico_pred)).reshape(1, -1)
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X_test)
        
        # Perform inference with the SVM meta-model
        final_prediction = self.svm.predict(X_test)
        final_probability = self.svm.predict_proba(X_test)
        
        return final_prediction[0], final_probability[0]

# Example usage
efficientnet = torch.load('/users/jrs596/scratch/dat/models/efficientnet_b0-IR.pth', map_location=lambda storage, loc: storage.cuda(0))

DisNet_pico = DisNet_picoIR(out_channels=4).to(device)
weights_path = "/users/jrs596/scratch/dat/models/DisNet_pico-IR_weights.pth"
weights = torch.load(weights_path, map_location=lambda storage, loc: storage.cuda(0))

# # Apply weights to the model
DisNet_pico.load_state_dict(weights)

svm_path = '/users/jrs596/scratch/dat/cross_val_predictions/svm_meta_model.joblib'

ensemble_classifier = EnsembleClassifier(efficientnet.eval(), DisNet_pico.eval(), svm_path)

dat_dir = '/users/jrs596/scratch/dat/IR_RGB_Comp_data/IR_split_1k/val'

# Prepare data
# img_size = config['input_size']

def load_data(dat_dir, img_size):
    data_transforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor()
    ])

    valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=1)
    return valloader
# image_path = 'path/to/your/image.jpg'
valloader = load_data(dat_dir, 455)
criterion = nn.CrossEntropyLoss()

#%%
my_metrics = toolbox.Metrics()
acc = 0
for i, (inputs, labels) in enumerate(valloader):
    
    image_tensor = inputs
    final_prediction, final_probability = ensemble_classifier.predict(image_tensor)
    out = torch.from_numpy(np.array(final_probability))     
    out = out.view(1, -1)

    loss = criterion(out, labels)
    _, preds = torch.max(out, 1)    

    stats = metrics.classification_report(labels.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
    stats_out = stats['weighted avg']
    my_metrics.update(loss, preds, labels, stats_out)
    # print(preds, labels)
    if preds == labels:
        acc += 1
acc = acc/len(valloader)
print(acc)
#%%
# Calculate metrics for the epoch
epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, f1_per_class = my_metrics.calculate()

print("Loss: {:.4f} Acc: {:.4f} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1))
    
# %%
