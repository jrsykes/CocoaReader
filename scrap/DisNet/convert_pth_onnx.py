import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile


model_path = '/local/scratch/jrs596/dat/models/CocoaNet18_quantised.pth'

torch_model = torch.jit.load(model_path)
torch_model.eval()

print(torch_model)