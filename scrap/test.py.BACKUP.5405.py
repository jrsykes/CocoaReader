<<<<<<< HEAD

import torch
from torchvision.models.convnext import _convnext, CNBlockConfig
import random

config = {'loss': 0, 'f1': 0, 'input_size': random.randint(250,350), 'one': random.randint(91,101), 'two': random.randint(187,197), 
              'three': random.randint(379,389), 'four': random.randint(763,773), 'five': random.randint(6,12)}

def build_model(num_classes, config):

    # # # Define your custom block settings
    my_block_settings = [
        CNBlockConfig(config['one'], config['two'], 3),
        CNBlockConfig(config['two'], config['three'], 3),
        CNBlockConfig(config['three'], config['four'], config['five']),
        CNBlockConfig(config['four'], None, 3)
    ]

=======
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

>>>>>>> cb7ec37ac214fa17ae1ab3330055a40ce1211709
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
<<<<<<< HEAD
        num_classes=1000,
=======
        num_classes=2,
>>>>>>> cb7ec37ac214fa17ae1ab3330055a40ce1211709
        block=None, 
        norm_layer=None
    )

    #model = models.convnext_tiny(weights = None)

<<<<<<< HEAD
    in_feat = model.classifier[2].in_features
    model.classifier[2] = torch.nn.Linear(in_feat, num_classes)
        
    return model #nn.DataParallel(model)

model = build_model(2, config)

print(model.summary())
=======
    #in_feat = model.classifier[2].in_features
    #model.classifier[2] = torch.nn.Linear(in_feat, num_classes)
        
    return model.to("cuda") #nn.DataParallel(model)




model = build_model(2, config)

#print model summary
summary(model, (3, config['input_size'], config['input_size']))
#printy last layer
print(model.classifier[2])

# %%
>>>>>>> cb7ec37ac214fa17ae1ab3330055a40ce1211709
