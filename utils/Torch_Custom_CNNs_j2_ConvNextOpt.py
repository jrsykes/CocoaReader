
from __future__ import print_function
from __future__ import division

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
import time
import copy
import wandb
from sklearn import metrics
from progress.bar import Bar
from torchvision.models.convnext import _convnext, CNBlockConfig
import argparse
import random
import json
from random_word import RandomWords

parser = argparse.ArgumentParser('encoder decoder examiner')
parser.add_argument('--model_name', type=str, default='test',
                        help='save name for model')
parser.add_argument('--project_name', type=str, default=None,
                        help='Name for wandb project')
parser.add_argument('--run_name', type=str, default=None,
                        help='Name for wandb run')
parser.add_argument('--sweep', action='store_true', default=False,
                        help='Run Waits and Biases optimisation sweep')
parser.add_argument('--sweep_id', type=str, default=None,
                        help='sweep if for weights and biases')
parser.add_argument('--sweep_config', type=str, default=None,
                        help='.yml sweep configuration file')
parser.add_argument('--sweep_count', type=int, default=100,
                        help='Number of models to train in sweep')
parser.add_argument('--root', type=str, default='/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat',
                        help='location of all data')
parser.add_argument('--data_dir', type=str, default='test',
                        help='location of all data')
parser.add_argument('--custom_pretrained', action='store_true', default=False,
                        help='Train useing specified pre-trained weights?')
parser.add_argument('--custom_pretrained_weights', type=str,
                        help='location of pre-trained weights')
parser.add_argument('--quantise', action='store_true', default=False,
                        help='Train with Quantization Aware Training?')
parser.add_argument('--batch_size', type=int, default=4,
                        help='Initial batch size')
parser.add_argument('--max_epochs', type=int, default=500,
                        help='n epochs before early stopping')
parser.add_argument('--min_epochs', type=int, default=10,
                        help='n epochs before loss is assesed for early stopping')
parser.add_argument('--patience', type=int, default=1,
                        help='n epochs to run without improvment in loss')
parser.add_argument('--beta', type=float, default=1.00,
                        help='minimum required per cent improvment in validation loss')
parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate, Default:1e-5')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Learning rate, Default:1e-5')
parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps, Default:1e-8')
parser.add_argument('--input_size', type=int, default=277,
                        help='image input size')
parser.add_argument('--alpha', type=float, default=1.4,
                        help='alpha for dynamic focal loss')
parser.add_argument('--gamma', type=float, default=1.6,
                        help='gamma for dynamic focal loss')
parser.add_argument('--arch', type=str, default='convnext_tiny',
                        help='Model architecture. resnet18, resnet50, resnext50, resnext101 or convnext_tiny')
parser.add_argument('--cont_train', action='store_true', default=False,
                        help='Continue training from previous checkpoint?')
parser.add_argument('--remove_batch_norm', action='store_true', default=False,
                        help='Deactivate all batchnorm layers?')
parser.add_argument('--split_image', action='store_true', default=False,
                        help='Split image into smaller chunks?')
parser.add_argument('--n_tokens', type=int, default=4,
                        help='Sqrt of number of tokens to split image into')


args = parser.parse_args()
print(args)

def setup(args):
    data_dir = os.path.join(args.root, args.data_dir)
    #Define some variable and paths
    os.environ['TORCH_HOME'] = os.path.join(args.root, "TORCH_HOME")
    
    data_dir = os.path.join(args.root, args.data_dir)
    num_classes= len(os.listdir(data_dir + '/train'))

    # Specify whether to use GPU or cpu. Quantisation aware training is not yet avalable for GPU.
   
    device = torch.device("cuda")
    
    ### Calculate and set bias for final layer based on imbalance in dataset classes
    dir_ = os.path.join(data_dir, 'train')
    list_cats = []
    for i in sorted(os.listdir(dir_)):
        _, _, files = next(os.walk(os.path.join(dir_, i)))
        list_cats.append(len(files))

    weights = []
    for i in list_cats:
        weights.append(np.log((max(list_cats)/i)))

    initial_bias = torch.FloatTensor(weights).to(device)

    return data_dir, num_classes, initial_bias, device


def train_model(model, optimizer, device, dataloaders_dict, criterion, patience, initial_bias, input_size, batch_size, n_tokens=None, AttNet=None, ANoptimizer=None):   
    since = time.time()
    val_loss_history = []
    best_f1 = 0.0
    best_f1_acc = 0.0
    
    #Save current weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts['module.fc.bias'] = initial_bias.to(device)

    epoch = 0
    #while patience >= 0: # Run untill validation loss has not improved for n epochs equal to patience variable and batchsize has decaed to 1
    while patience > 0 and epoch < args.max_epochs:
        print('\nEpoch {}'.format(epoch))
        print('-' * 10) 
        step  = 0
        #Ensure minimum number of epochs is met before patience is allow to reduce
        if len(val_loss_history) > args.min_epochs:
           #If the current loss is not at least 0.5% less than the lowest loss recorded, reduce patiece by one epoch
            if val_loss_history[-1] > min(val_loss_history)*args.beta:
                patience -= 1
            elif val_loss_history[-1] == np.nan:
                patience -= 1
            else:
               #If validation loss improves by at least 0.5%, reset patient to initial value
               patience = args.patience
        print('Patience: ' + str(patience) + '/' + str(args.patience))
       #If patience gets to 6, half batch size, reintialise dataloader and revert to best model weights to undo any overfitting.
       #Do this for every epoch where patience is less than 6. i.e. if inital batch size = 128, the last epoch will have a batchsize of at most 2.

       #Training and validation loop
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                if AttNet != None:
                    AttNet.train()
 
            elif phase == 'val':
               #Model quatisation with quantisation aware training
                model.eval()   # Set model to evaluate mode
                if AttNet != None:
                    AttNet.eval()

            running_loss = 0.0
            running_corrects = 0
            running_precision = 0
            running_recall = 0
            running_f1 = 0  
           # Iterate over data.
           #Get size of whole dataset split
            n = len(dataloaders_dict[phase].dataset)
           #Begin training
            print(phase)
            #n_steps_per_epoch = len(dataloaders_dict['train'].dataset)/batch_size
            with Bar('Learning...', max=n/batch_size+1) as bar:
               
                for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):

                    #split images in bacth into 16 non-overlapping chunks then recombine into bacth
                    if args.split_image == True:
                        
                        #split batch into list of tensors
                        inputs = torch.split(inputs, 1, dim=0)
                        #print(inputs[0].shape)
                        token_size = int(input_size / n_tokens)
                        x, y, h ,w = 0, 0, token_size, token_size
                        ims = []
                        for t in inputs:
                            #crop im to 16 non-overlapping 277x277 tensors
                            for i in range(n_tokens):
                                for j in range(n_tokens):
                                    t.squeeze_(0)
                                    im1 = t[:, x:x+h, y:y+w]
                                    ims.append(im1)
                                    y += w
                                y = 0
                                x += h
                            x, y, h ,w = 0, 0, token_size, token_size
                        #convert list of tensors into a batch tensor of shape (512, 3, 277, 277)
                        inputs = torch.stack(ims, dim=0)
                    
                    #Load images and lables from current batch onto GPU(s)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()       
                    if ANoptimizer != None:
                        ANoptimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                       # Get model outputs and calculate loss
                       # In train mode we calculate the loss by summing the final output and the auxiliary output
                       # but in testing we only consider the final output.
                        outputs = model(inputs)

                        #use PCA to reduce dimensionality of output to 32x2
                        if args.split_image == True:
                            #old_output_shell = outputs[0:batch_size,0:num_classes]
                            new_batch = []
                            for i in range(batch_size):
                                outputs_ = outputs[i*n_tokens**2:(i+1)*n_tokens**2]
                                outputs_ = outputs_.detach().cpu()
                                
                                outputs_flat = torch.empty(0)
                                for i in range(outputs_.shape[1]):
                                    outputs_flat = torch.cat((outputs_flat,outputs_[:,i]),0)

                                outputs_ = outputs_flat.to(device).unsqueeze(0)
                                outputs_ = AttNet(outputs_)
                                                        
                                new_batch.append(outputs_)

                            outputs = torch.stack(new_batch, dim=0).squeeze(1)

                        loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)    
                        stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
                        stats_out = stats['weighted avg']
                       
                  
                       # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()  
                            if ANoptimizer != None:
                                ANoptimizer.step() 
      

                   #Calculate statistics
                   #Here we multiply the loss and other metrics by the number of lables in the batch and then divide the 
                   #running totals for these metrics by the total number of training or validation samples. This controls for 
                   #the effect of batch size and the fact that the size of the last batch will less than args.batch_size
                    running_loss += loss.item() * args.batch_size # inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data) 
                    running_precision += stats_out['precision'] * args.batch_size # inputs.size(0)
                    running_recall += stats_out['recall'] * args.batch_size # inputs.size(0)
                    running_f1 += stats_out['f1-score'] * args.batch_size # inputs.size(0)    

                    bar.next()  
           #Calculate statistics for epoch
            n = len(dataloaders_dict[phase].dataset)
            epoch_loss = float(running_loss / n)
            epoch_acc = float(running_corrects.double() / n)
            epoch_precision = (running_precision) / n         
            epoch_recall = (running_recall) / n        
            epoch_f1 = (running_f1) / n 
            AIC_ = AIC(model=model, train_loader=dataloaders_dict['train'], loss=epoch_loss)
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, epoch_precision, epoch_recall, epoch_f1))
            # Save statistics to tensorboard log
            

           # Save model and update best weights only if recall has improved
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_f1_acc = epoch_acc
                best_f1_loss = epoch_loss
                best_f1_AIC = AIC_
                best_model_wts = copy.deepcopy(model.state_dict())  
                model_out = model
    
                #PATH = os.path.join(args.root, 'models', args.model_name)
                #torch.save(model.module, PATH + '.pth') 
  
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            
            if phase == 'train':
                
                wandb.log({"epoch": epoch, "Train_loss": epoch_loss, "Train_acc": epoch_acc, "Train_F1": epoch_f1, "AIC": AIC_})  
            else:
                wandb.log({"epoch": epoch, "Val_loss": epoch_loss, "Val_acc": epoch_acc, "Val_F1": epoch_f1, "Best_F1": best_f1, "Best_F1_acc": best_f1_acc})


        bar.finish() 
        epoch += 1
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Acc of saved model: {:4f}'.format(best_f1_acc))
    print('F1 of saved model: {:4f}'.format(best_f1))
    return model_out, best_f1, best_f1_loss, best_f1_AIC

def AIC(model, train_loader, loss):
    #get number of model peramaters from the model
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)   
    #size of training dataset
    N = len(train_loader.dataset)
    AIC_ = -2/N*loss+2*k/N
    return AIC_

def build_datasets(input_size, data_dir):
    # Data augmentation and normalization for training
    # Just normalization for device
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomCrop(input_size, pad_if_needed=True, padding_mode = 'reflect'),
            transforms.Resize((input_size,input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor(),
        ]),
    }   

    print("Initializing Datasets and Dataloaders...")
    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    return image_datasets


  

def Remove_module_from_layers(unpickled_model_wts):
    new_keys = []
    for key, value in unpickled_model_wts.items():
        new_keys.append(key.replace('module.', ''))
    for i in new_keys:
        unpickled_model_wts[i] = unpickled_model_wts.pop('module.' + i)
    return unpickled_model_wts

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
        num_classes=num_classes,
        block=None, 
        norm_layer=None
    )
  
    return model #nn.DataParallel(model)

def set_batchnorm_momentum(self, momentum):
    for m in self.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = momentum
    return self

class AttentionNet(nn.Module):
    def __init__(self, num_classes, num_tokens):
        super(AttentionNet, self).__init__()
        num_heads = num_classes
        embed_dim = num_tokens**2*num_classes
        head_dim = embed_dim//num_heads
        # define linear transformations for queries, keys, and values
        self.query_transform = nn.Linear(embed_dim, num_heads * head_dim)
        self.key_transform = nn.Linear(embed_dim, num_heads * head_dim)
        self.value_transform = nn.Linear(embed_dim, num_heads * head_dim)
        #apply muti-AttentionNet head
        self.Attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        #apply layer normalisation
        self.layernorm = nn.LayerNorm(embed_dim)
        #apply linear layer
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_classes)


    def forward(self, x):
        #apply AttentionNet
        queries = self.query_transform(x)
        keys = self.key_transform(x)
        values = self.value_transform(x)
        x, _ = self.Attention(queries, keys, values)
        x = self.fc1(x)
        #apply layer normalisation
        x = self.layernorm(x)
        #apply linear layer
        x = self.fc2(x)
        #apply relu activation
        x = F.softmax(x, dim=1)
        return x


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
        for i in range(inputs.size(0)):
            #get filename from dataset
            filename = self.dataloader.dataset.samples[step + i][0].split("/")[-1]
            if filename not in self.weights_dict:
                self.weights_dict[filename] = 1
            if preds[i] != targets[i]:
                self.weights_dict[filename] += self.delta
        step += inputs.size(0)
        
        # Apply weights to loss based on weights_dict
        weighted_loss = torch.zeros(1).to(loss.device)
        for filename, weight in self.weights_dict.items():
            if weight > 1:
                weighted_loss += loss * weight
            else:
                weighted_loss += loss
        weighted_loss /= len(self.weights_dict)

        return weighted_loss, step


def train(config):

    data_dir, num_classes, initial_bias, device = setup(args)

    
    num_classes = len(os.listdir(os.path.join(data_dir, 'train')))
    model_ft = build_model(num_classes=num_classes, config=config)

    model_ft = model_ft.to(device)

    
    optimizer = torch.optim.AdamW(model_ft.parameters(), lr=args.learning_rate,
                                            weight_decay=args.weight_decay, eps=args.eps)
    

    image_datasets = build_datasets(input_size=config['input_size'], data_dir=data_dir)
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True) for x in ['train', 'val']}
    
    criterion = nn.CrossEntropyLoss()

    trained_model, best_f1, best_f1_loss = train_model(model=model_ft, optimizer=optimizer, device=device, dataloaders_dict=dataloaders_dict, criterion=criterion, patience=args.patience, initial_bias=initial_bias, input_size=config['input_size'], n_tokens=args.n_tokens, batch_size=args.batch_size, AttNet=None, ANoptimizer=None)
    
    return trained_model, best_f1, best_f1_loss

def sample_from_log_distribution(min_val, max_val, size=1, base=10):
    """
    Generate random samples from a log distribution between min_val and max_val.

    Args:
        min_val (float): Minimum value of the log distribution.
        max_val (float): Maximum value of the log distribution.
        size (int, optional): Number of random samples to generate. Defaults to 1.
        base (float, optional): Base of the logarithm. Defaults to 10.

    Returns:
        numpy.ndarray: An array of random samples from the log distribution.
    """
    log_min = np.log(min_val) / np.log(base)
    log_max = np.log(max_val) / np.log(base)
    log_samples = np.random.uniform(log_min, log_max, size=size)
    return base ** log_samples

def main():

    if args.run_name is None:
        run_name = RandomWords().get_random_word() + '_run_' + str(time.time())[-2:]
        wandb.init(project=args.project_name, name=run_name)
    else:
        wandb.init(project=args.project_name, name=args.run_name)
        run_name = args.run_name

    #stochastic_depth_prob = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    config = {'loss': 0, 'f1': 0, 'input_size': 320, 'one': random.randint(91,101), 'two': random.randint(187,197), 
              'three': random.randint(379,389), 'four': random.randint(763,773), 'five': random.randint(6,12)}
    #config = {'loss': 0, 'f1': 0, 'input_size': 330, 'one': 96, 'two': 192, 
     #          'three': 384, 'four': 768, 'five': 9}


    trained_model, best_f1, best_f1_loss, best_f1_AIC = train(config)
  
    n_parameters = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)   

    config['n_parameters'] = n_parameters
    config['loss'] = best_f1_loss
    config['f1'] = best_f1
    config['AIC'] = best_f1_AIC

    #make directory if it doesn't exist
    os.makedirs(os.path.join("/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/models/HypSweep/", args.project_name), exist_ok=True)
    out_file = os.path.join("/jmain02/home/J2AD016/jjw02/jjs00-jjw02/dat/models/HypSweep/", args.project_name, run_name + ".json")
    
    
    # Save the updated dictionary back to the JSON file
    with open(out_file, 'w') as f:
        json.dump(config, f)

        
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main()
