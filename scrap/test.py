import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch

num_classes = 2

model = models.resnet18(weights=None)
in_feat = model.fc.in_features
model.fc = nn.Linear(in_feat, num_classes)

#for key, value in model_ft.items():
#    print(key)
#    print(value)


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


x = torch.randn(10, 3, 24, 24)
output1 = model.bn1(x)
output2 = model(x)
print(torch.allclose(output1, output2))
exit()
model.apply(deactivate_batchnorm)

output1 = model[0](x)
output2 = model(x)
torch.allclose(output1, output2)
