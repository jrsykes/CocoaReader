#%%
from torchvision import datasets, transforms, models
import torch

model_ft = models.convnextv2_atto(weights = None)


#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)


print(model_ft)


#%%
print('HELLO')
# %%
