#%%


import torch
import torch.nn as nn

class DynamicFocalLoss(nn.Module):
    def __init__(self, delta=1, dataloader=None):
        super(DynamicFocalLoss, self).__init__()
        self.delta = delta
        self.dataloader = dataloader
        self.weights_dict = {}

    def forward(self, inputs, targets, step):
        loss = nn.CrossEntropyLoss()(inputs, targets)
        # Update weights_dict based on targets and predictions
        preds = torch.argmax(inputs, dim=1)
        batch_weight = 0
        for i in range(inputs.size(0)):
            #get filename from dataset
            filename = self.dataloader.dataset.samples[step + i][0].split("/")[-1]
            if filename not in self.weights_dict:
                self.weights_dict[filename] = 1
            if preds[i] != targets[i]:
                self.weights_dict[filename] += self.delta
    
            weight = self.weights_dict[filename]
            if weight > 1:
                batch_weight += weight
        
        loss *= batch_weight
        step += inputs.size(0)
 
        return loss, step

#%%