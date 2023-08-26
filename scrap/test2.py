#%%
import torch.nn as nn
import torch

criterion = nn.CrossEntropyLoss()

#%%

out = torch.tensor([[0, 0, 0, 10]], dtype=torch.float32)
lable = torch.tensor([1], dtype=torch.float32)

loss = criterion(out, lable)

print(loss)


#%%