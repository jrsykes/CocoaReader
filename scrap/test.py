import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch

num_classes = 2

model = models.convnext_tiny(weights=None)

#for key, value in model_ft.items():
#    print(key)
#    print(value)


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


#model_ft.apply(deactivate_batchnorm)
print(model)
exit()
output1 = model[0](x)
output2 = model(x)
torch.allclose(output1, output2)
