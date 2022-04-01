
import torch.nn.functional as F
import torch.nn as nn
import torch

y_true = torch.tensor([[1,1,1,1],[0.5,0.5,0.5,0.5]])

y_pred = torch.tensor([[1.6,2.1,0.5,1.3,],[0.2,1.1,0.1,0.4]])

#print(y_true)
#print(y_pred)


kl_loss = nn.KLDivLoss(reduction="batchmean")
# input should be a distribution in the log space
input = F.log_softmax(torch.randn(2, 5, requires_grad=True))
# Sample a batch of distributions. Usually this would come from the dataset
target = F.softmax(torch.rand(2, 5))


output = kl_loss(input, target)

print(target.shape)

#log_target = F.log_softmax(torch.rand(3, 5))
#output = kl_loss(input, log_target, log_target=True)


