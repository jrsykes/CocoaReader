#%%
import torch
from torchvision.models.convnext import _convnext, CNBlockConfig
import random
from torchsummary import summary
#%%

config = {'loss': 0, 'f1': 0, 'input_size': 330, 
          'one': 91, 'two': 187, 
        'three': 379, 'four': 763, 
        'five': 6}

def build_model(num_classes, config):

    # # # Define your custom block settings
    my_block_settings = [
        CNBlockConfig(config['one'], config['two'], 3),
        CNBlockConfig(config['two'], config['three'], 3),
        CNBlockConfig(config['three'], config['four'], config['five']),
        CNBlockConfig(config['four'], None, 3)
    ]

    # Define a new function that calls _convnext() with the modified arguments
    def my_convnext(block_setting, *args, **kwargs):
        kwargs['block_setting'] = block_setting
        return _convnext(*args, **kwargs)

    # Call my_convnext with your custom block settings
    model = my_convnext(
        block_setting=my_block_settings, 
        stochastic_depth_prob=0.1,
        weights = None,
        progress=False,
        layer_scale=1e-6,
        num_classes=2,
        block=None, 
        norm_layer=None
    )

    #model = models.convnext_tiny(weights = None)

    #in_feat = model.classifier[2].in_features
    #model.classifier[2] = torch.nn.Linear(in_feat, num_classes)
        
    return model.to("cuda") #nn.DataParallel(model)




model = build_model(2, config)

#print model summary
summary(model, (3, config['input_size'], config['input_size']))
#printy last layer
print(model.classifier[2])

# %%
