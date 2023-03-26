#%%
from torchvision import datasets, models
from torchvision.models import ConvNeXt_Tiny_Weights


model_ft = models.convnext_tiny(weights = ConvNeXt_Tiny_Weights.DEFAULT)


print(model_ft)

#%%