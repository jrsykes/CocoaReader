#%%
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
#import models
from torchbearer import Trial
import cv2
import torch.nn.functional as F
import os
from torchvision import datasets, transforms
import sys
sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')
from ArchitectureZoo import DisNet_pico




n_imgs = 8
#%%

class ResNet_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(ResNet_CAM, self).__init__()
        self.resnet = net
        convs = nn.Sequential(*list(net.children())[:-1])
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = nn.Sequential(*list(net.children())[-1:])
        
    def forward(self, x):
        x = self.first_part_conv(x.to('cuda'))
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view((1, -1))
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x.to('cuda'))
    
#%%


class EfficientNet_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(EfficientNet_CAM, self).__init__()
        self.efficientnet = net
        convs = nn.Sequential(*list(net.children())[:-1])
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:-1]  # Exclude the last layer
        self.linear = nn.Sequential(convs[-1], *list(net.children())[-1:])  # Include the last layer in linear

        
    def forward(self, x):
        x = self.first_part_conv(x.to('cuda'))
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = F.adaptive_avg_pool2d(x, (1280, 1280))
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x.to('cuda'))


#%%

class DisNet_nano_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(DisNet_nano_CAM, self).__init__()
        self.disnet_pico = net
        convs = nn.Sequential(net.conv1, net.cnblock1, net.pool, net.conv2, net.cnblock2, net.pool)
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = nn.Sequential(net.fc1, net.fc2, net.fc3)
        
    def forward(self, x):
        x = self.first_part_conv(x.to('cuda'))
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = torch.flatten(x, 1) # flatten before feeding to linear layers
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x.to('cuda'))

class DisNet_pico_CAM(nn.Module):
    def __init__(self, net, layer_k):
        super(DisNet_pico_CAM, self).__init__()
        self.disnet_pico = net
        convs = nn.Sequential(net.conv1, net.cnblock1, net.conv2, net.cnblock2, net.pool, net.conv3, net.cnblock3, net.pool)
        self.first_part_conv = convs[:layer_k]
        self.second_part_conv = convs[layer_k:]
        self.linear = nn.Sequential(net.fc1, net.fc2, net.fc3)
        
    def forward(self, x):
        x = self.first_part_conv(x.to('cuda'))
        x.register_hook(self.activations_hook)
        x = self.second_part_conv(x)
        x = torch.flatten(x, 1) # flatten before feeding to linear layers
        x = self.linear(x)
        return x
    
    def activations_hook(self, grad):
        self.gradients = grad
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.first_part_conv(x.to('cuda'))



# %%

def superimpose_heatmap(heatmap, img):
    resized_heatmap = cv2.resize(heatmap.numpy(), (img.shape[2], img.shape[3]))
    resized_heatmap = np.uint8(255 * resized_heatmap)
    resized_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)
    superimposed_img = torch.Tensor(cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2RGB)) * 0.006 + img[0].permute(1,2,0)
    
    return superimposed_img

def get_grad_cam(net, img):
    net.eval()
    pred = net(img)
    pred[:,pred.argmax(dim=1)].backward()
    gradients = net.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    activations = net.get_activations(img).detach()
    for i in range(activations.size(1)):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze().detach().to('cpu')
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)
    
    return torch.Tensor(superimpose_heatmap(heatmap, img).permute(2,0,1))

#%%

# resnet = torch.load('/local/scratch/jrs596/dat/models/resnet18-IR.pth')
DisNet_pico = torch.load('/local/scratch/jrs596/dat/models/DisNet_pico_IR.pth', map_location=lambda storage, loc: storage.cuda(0))
# efficientnet = torch.load('/local/scratch/jrs596/dat/models/efficientnet-IR.pth')
DisNet_nano = torch.load('/local/scratch/jrs596/dat/models/DisNet_nano_IR.pth', map_location=lambda storage, loc: storage.cuda(0))



#%%Resnet18 IR

# layer_k = 9

# imgs = torch.Tensor(layer_k, n_imgs, 3, img_size, img_size)
# it = iter(valloader)
# for i in range(0,n_imgs):
#     img, _ = next(it)
#     imgs[0][i] = img[0]
#     for k in range(1,layer_k):
#         resnet_cam_net = ResNet_CAM(resnet, k)
#         imgs[k][i] = get_grad_cam(resnet_cam_net, img)



# torchvision.utils.save_image(imgs.view(-1, 3, img_size, img_size), "gradcam_resnet18-IR" + ".png",nrow=n_imgs, pad_value=1)
#%%EfficientNet IR

# layer_k = 5

# imgs = torch.Tensor(layer_k, n_imgs, 3, img_size, img_size)
# it = iter(valloader)
# for i in range(0,n_imgs):
#     img, _ = next(it)
#     imgs[0][i] = img[0]
#     for k in range(1,layer_k):
#         efficientnet_cam_net = EfficientNet_CAM(efficientnet, k)
#         imgs[k][i] = get_grad_cam(efficientnet_cam_net, img)



# torchvision.utils.save_image(imgs.view(-1, 3, img_size, img_size), "gradcam_efficientnet-IR" + ".png",nrow=n_imgs, pad_value=1)
#%%DisNet-pico IR
img_size = 494
data_transforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
                transforms.ToTensor()])

dat_dir = '/local/scratch/jrs596/dat/GradCam_imgs_IR'
valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=6)

layer_k = [1,2,3,4,6,7]
imgs = torch.Tensor(len(layer_k)+1, n_imgs, 3, img_size, img_size)
it = iter(valloader)
for i in range(0,n_imgs):
    img, _ = next(it)
    imgs[0][i] = img[0]
    #for k in range(1,layer_k):
    pass_ = 1
    for k in layer_k:
        DisNet_cam_net = DisNet_pico_CAM(DisNet_pico, k)
        imgs[pass_][i] = get_grad_cam(DisNet_cam_net, img)
        pass_ += 1


torchvision.utils.save_image(imgs.view(-1, 3, img_size, img_size), "gradcam_DisNet-pico-IR" + ".png",nrow=n_imgs, pad_value=1)

#%%DisNet-nano IR
img_size = 478
data_transforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
                transforms.ToTensor()])

dat_dir = '/local/scratch/jrs596/dat/GradCam_imgs_IR'
valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=6)

layer_k = [1,2,4,5]

imgs = torch.Tensor(len(layer_k)+1, n_imgs, 3, img_size, img_size)
it = iter(valloader)
for i in range(0,n_imgs):
    img, _ = next(it)
    imgs[0][i] = img[0]
    #for k in range(1,layer_k):
    pass_ = 1
    for k in layer_k:
        DisNet_cam_net = DisNet_nano_CAM(DisNet_nano, k)
        imgs[pass_][i] = get_grad_cam(DisNet_cam_net, img)
        pass_ += 1



torchvision.utils.save_image(imgs.view(-1, 3, img_size, img_size), "gradcam_disnet-nano-IR" + ".png",nrow=n_imgs, pad_value=1)


#%% Get model sizes
# resnet = torch.load('/local/scratch/jrs596/dat/models/resnet18_RGB.pth')
# #print n params
# num_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
# print("ResNet18 n params: {:,}".format(num_params))

DisNet_pico = torch.load('/local/scratch/jrs596/dat/models/DisNet_pico_RGB.pth', map_location=lambda storage, loc: storage.cuda(0))
#print n params
num_params = sum(p.numel() for p in DisNet_pico.parameters() if p.requires_grad)
print("DisNet_pico n params: {:,}".format(num_params))


DisNet_nano = torch.load('/local/scratch/jrs596/dat/models/DisNet_nano_RGB.pth', map_location=lambda storage, loc: storage.cuda(0))
#print n params
num_params = sum(p.numel() for p in DisNet_nano.parameters() if p.requires_grad)
print("DisNet_nano n params: {:,}".format(num_params))


# efficientnet = torch.load('/local/scratch/jrs596/dat/models/efficientnet_RGB.pth')
# #print n params
# num_params = sum(p.numel() for p in efficientnet.parameters() if p.requires_grad)
# print("efficientnet n params: {:,}".format(num_params))


# #%%
# # %%ResNet RGB

# layer_k = 9

# imgs = torch.Tensor(layer_k, n_imgs, 3, img_size, img_size)
# it = iter(valloader)
# for i in range(0,n_imgs):
#     img, _ = next(it)
#     imgs[0][i] = img[0]
#     for k in range(1,layer_k):
#         resnet_cam_net = ResNet_CAM(resnet, k)
#         imgs[k][i] = get_grad_cam(resnet_cam_net, img)



# torchvision.utils.save_image(imgs.view(-1, 3, img_size, img_size), "gradcam_ResNet18-RGB" + ".png",nrow=n_imgs, pad_value=1)
# #%%EfficientNet RGB

# layer_k = 5

# imgs = torch.Tensor(layer_k, n_imgs, 3, img_size, img_size)
# it = iter(valloader)
# for i in range(0,n_imgs):
#     img, _ = next(it)
#     imgs[0][i] = img[0]
#     for k in range(1,layer_k):
#         efficientnet_cam_net = EfficientNet_CAM(efficientnet, k)
#         imgs[k][i] = get_grad_cam(efficientnet_cam_net, img)



#torchvision.utils.save_image(imgs.view(-1, 3, img_size, img_size), "gradcam_EfficientNet-RGB" + ".png",nrow=n_imgs, pad_value=1)


#%%DisNet-pico RGB
img_size = 494
data_transforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
                transforms.ToTensor()])

dat_dir = '/local/scratch/jrs596/dat/GradCam_imgs_RGB'
valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=6)


layer_k = [1,2,3,4,6,7]
imgs = torch.Tensor(len(layer_k)+1, n_imgs, 3, img_size, img_size)
it = iter(valloader)
for i in range(0,n_imgs):
    img, _ = next(it)
    imgs[0][i] = img[0]
    pass_ = 1
    for k in layer_k:
        DisNet_cam_net = DisNet_pico_CAM(DisNet_pico, k)
        imgs[pass_][i] = get_grad_cam(DisNet_cam_net, img)
        pass_ += 1



torchvision.utils.save_image(imgs.view(-1, 3, img_size, img_size), "gradcam_DisNet-pico-RGB" + ".png",nrow=n_imgs, pad_value=1)

#%%DisNet-nano RGB
img_size = 478
data_transforms = transforms.Compose([
        transforms.Resize((img_size,img_size)),
                transforms.ToTensor()])

dat_dir = '/local/scratch/jrs596/dat/GradCam_imgs_RGB'
valset = datasets.ImageFolder(os.path.join(dat_dir), data_transforms)
valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=6)

layer_k = [1,2,4,5]
imgs = torch.Tensor(len(layer_k)+1, n_imgs, 3, img_size, img_size)
it = iter(valloader)
for i in range(0,n_imgs):
    img, _ = next(it)
    imgs[0][i] = img[0]
    pass_ = 1
    for k in layer_k:
        DisNet_cam_net = DisNet_nano_CAM(DisNet_nano, k)
        imgs[pass_][i] = get_grad_cam(DisNet_cam_net, img)
        pass_ += 1



torchvision.utils.save_image(imgs.view(-1, 3, img_size, img_size), "gradcam_DisNet-nano-RGB" + ".png",nrow=n_imgs, pad_value=1)

#%%
