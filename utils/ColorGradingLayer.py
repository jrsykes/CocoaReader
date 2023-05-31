#%%
import torch
from torchvision import models
#from torchvision.models import ResNet18_Weights

class CrossTalkColorGrading(torch.nn.Module):
    def __init__(self, matrix=None):
        super().__init__()
        if matrix == 'Best':
            matrix = torch.tensor([[-2.569969892501831055e-01,2.154911756515502930e-01,1.519576273858547211e-02],
                [2.386483430862426758e+00,-3.911899626255035400e-01,-1.561194300651550293e+00],
                [-6.149564385414123535e-01,1.582644462585449219e+00,9.494693279266357422e-01]])
        elif matrix is None:
            #random matrix
            matrix = torch.rand(3,3)
        
        self.matrix = torch.nn.Parameter(matrix)

    def forward(self, img):
        return self._transform(img)

    def _transform(self, img):
        img_tensor = img.permute(0, 2, 3, 1)
        img_tensor = img_tensor @ self.matrix
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        img_tensor = torch.clamp(img_tensor, 0, 1)
        return img_tensor

class CGResNet18(torch.nn.Module):
    def __init__(self, num_classes, matrix=None):
        super().__init__()
        self.color_grading = CrossTalkColorGrading(matrix=matrix)
        self.resnet18 = models.resnet18(weights=None)
        # Modify the last layer of ResNet18 to have the desired number of output nodes
        self.resnet18.fc = torch.nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        x = self.color_grading(x)
        x = self.resnet18(x)
        return x
