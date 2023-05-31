#%%
import sys
import torch
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms import ToTensor, ToPILImage
from matplotlib import pyplot as plt
import os
import numpy as np
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
from ColorGradingLayer import CGResNet18, CrossTalkColorGrading

#%%
#model_path = "/local/scratch/jrs596/dat/IR_RGB_Comp_data/cross-val_models"

# for model_pth in os.listdir(model_path):
#     if model_pth.startswith("IR"):
#         #load saved model
#         model = torch.load(os.path.join(model_path, model_pth))
    
#         #print(model.color_grading.matrix)

#         #Image load img
#         img_path = "/local/scratch/jrs596/dat/IR_RGB_Comp_data/cross-val_IR/fold_0/val/FPR/M3130012.JPG"
#         img = Image.open(img_path)
#         #convert img to tensor
#         input_tensor = ToTensor()(img)
#         input_batch = input_tensor.unsqueeze(0).to('cuda')  # Create a mini-batch as expected by the model

#         # Apply the CrossTalkColorGrading layer to the image
#         with torch.no_grad():
#             color_graded = model.color_grading(input_batch)

#         #convert tensor to PIL image
#         color_graded = ToPILImage()(color_graded.squeeze(0).to('cpu'))

#         #save the image
#         color_graded.save("/local/scratch/jrs596/dat/IR_RGB_Comp_data/transformed_IR_images/" + model_pth + ".JPG")

# %%

matrix = torch.tensor([[-2.569969892501831055e-01,2.154911756515502930e-01,1.519576273858547211e-02],
    [2.386483430862426758e+00,-3.911899626255035400e-01,-1.561194300651550293e+00],
    [-6.149564385414123535e-01,1.582644462585449219e+00,9.494693279266357422e-01]])
            
model = CGResNet18(num_classes=4, matrix=matrix)

#Image load img
#img_path = "/local/scratch/jrs596/dat/IR_RGB_Comp_data/cross-val_IR/fold_0/val/FPR/M3130012.JPG"
root = '/local/scratch/jrs596/dat/IR_RGB_Comp_data/compiled_IR'
for class_ in os.listdir(root):
    #save the image
    dest = os.path.join('/local/scratch/jrs596/dat/IR_RGB_Comp_data/ColorGradedImages', class_)
    os.makedirs(dest, exist_ok=True)

    for img_pth in os.listdir(os.path.join(root, class_)):
        img = Image.open(os.path.join(root, class_, img_pth))
        #convert img to tensor
        input_tensor = ToTensor()(img)
        input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the mode
        # Apply the CrossTalkColorGrading layer to the image
        with torch.no_grad():
            color_graded = model.color_grading(input_batch)
        #convert tensor to PIL image
        color_graded = ToPILImage()(color_graded.squeeze(0))
        
        #save
        color_graded.save(os.path.join(dest, img_pth))

#convert tensor to np array
#color_graded = color_graded.squeeze(0).numpy()
#color_graded = np.transpose(color_graded, (1, 2, 0))  # Now 'image' has shape (287, 287, 3)

#show np array as image
#plt.imshow(color_graded)
# %%

sweep = 'devoted-sweep-49'

matrix_pth = "/local/scratch/jrs596/dat/IR_RGB_Comp_data/best_matrix_sweep/" + sweep + "_matrix.csv"
#load torch tensor from csv
matrix = torch.from_numpy(np.loadtxt(matrix_pth, delimiter=','))
#format float
matrix = matrix.float()
print(matrix)

#matrix = torch.tensor([[-2.569969892501831055e-01,2.154911756515502930e-01,1.519576273858547211e-02],
 #   [2.386483430862426758e+00,-3.911899626255035400e-01,-1.561194300651550293e+00],
  #  [-6.149564385414123535e-01,1.582644462585449219e+00,9.494693279266357422e-01]])
            
model = CGResNet18(num_classes=4, matrix=matrix)

#Image load img
#img_path = "/local/scratch/jrs596/dat/IR_RGB_Comp_data/cross-val_IR/fold_0/val/FPR/M3130012.JPG"
root = '/local/scratch/jrs596/dat/IR_RGB_Comp_data/compiled_IR/FPR/M3140028.JPG'
#save the image
dest = os.path.join('/local/scratch/jrs596/dat/IR_RGB_Comp_data/demo_ColourGrading', sweep  + '.JPG')

img = Image.open(root)
#convert img to tensor
input_tensor = ToTensor()(img)
input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the mode
# Apply the CrossTalkColorGrading layer to the image
with torch.no_grad():
    color_graded = model.color_grading(input_batch)
#convert tensor to PIL image
color_graded = ToPILImage()(color_graded.squeeze(0))

#save
color_graded.save(dest)

#%%