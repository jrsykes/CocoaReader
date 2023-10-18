from __future__ import print_function
from __future__ import division


import torch
import numpy as np
import time
# import copy
import wandb
# from sklearn import metrics
from progress.bar import Bar
import os
#import sys
from random_word import RandomWords
import toolbox
#sys.path.append(os.path.join(os.getcwd(), 'scripts/CocoaReader/utils'))
# from toolbox import DynamicFocalLoss
import torch.nn.functional as F
import pandas as pd
from itertools import combinations


def train_model(args, model, optimizer, device, dataloaders_dict, criterion, patience, batch_size, num_classes):      
    # @torch.compile
    # def run_model(x):
    #     return model(x)
        
    #Check environmental variable WANDB_MODE
    if args.WANDB_MODE == 'offline':   
        if args.sweep_config == None:
            if args.run_name is None:
                run_name = RandomWords().get_random_word() + '_' + str(time.time())[-2:]
                wandb.init(project=args.project_name, name=run_name, mode="offline")
            else:
                wandb.init(project=args.project_name, name=args.run_name, mode="offline")
                run_name = args.run_name
        else:
            run_name = wandb.run.name
    else:
        if args.sweep_config == None:
            if args.run_name is None:
                run_name = RandomWords().get_random_word() + '_' + str(time.time())[-2:]
                wandb.init(project=args.project_name, name=run_name)
            else:
                wandb.init(project=args.project_name, name=args.run_name)
                run_name = args.run_name
        else:
            run_name = wandb.run.name
    
    my_metrics = toolbox.Metrics(metric_names= ['loss'], num_classes=num_classes)

    since = time.time()
    val_loss_history = []
    best_loss = 10000.0
    
    epoch = 0
    # Run untill validation loss has not improved for n epochs=patience or max_epochs is reached
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
   
       #Training and validation loop
        for phase in ['train', 'val']:
            encoded_images_lst = []
            if phase == 'train':
                model.train()  # Set model to training mode
 
                model.eval()   # Set model to evaluate mode

           #Get size of whole dataset split
            n = len(dataloaders_dict[phase].dataset)
           #Begin training
            print(phase)
            with Bar('Learning...', max=n/batch_size+1) as bar:
               
                for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
                    
                    #Load images and lables from current batch onto GPU(s)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()       

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                       # Get model outputs and calculate loss
                       # In train mode we calculate the loss by summing the final output and the auxiliary output
                       # but in testing we only consider the final output.
                                              
                        #compress inputs to 240 x 240
                        
                        SRinputs = inputs
                        inputs = F.interpolate(inputs, size=(240, 240), mode='bilinear', align_corners=True)

                        encoded, decoded  = model(inputs)
                     
                        #read csv 
                        distance_df = pd.read_csv('/scratch/staff/jrs596/dat/FAIGB/DisNet_TaxonomyMatrix.csv', header=0)                     
                        
                        class_list = list(distance_df.columns)
                        
                        encoded_images_lst.extend([(encoded[i], class_list[i]) for i in range(encoded.size(0))])


                        # dist = distance_df.iloc[encoded_images_lst[0][1].item(), encoded_images_lst[1][1].item()]
                        # print()
                        # #print label pair
                        # print(encoded_images_lst[0][1].item(), encoded_images_lst[1][1].item())
                        # #latent space pair
                        # print(encoded_images_lst[0][0].size(), encoded_images_lst[1][0].size())
                        # #print distance
                        # print(dist)
                        
                        
    
                        contrastive_loss = toolbox.average_contrastive_loss(encoded_images_lst, distance_df)
                        print()
                        print(contrastive_loss.size())
                        exit()
                                                
                        loss = criterion(decoded, SRinputs)

                        
                        
                        
                        l1_norm = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
                        
                        loss += args.l1_lambda * l1_norm                     
                                               
                  
                       # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()  
                           
                        #Update metrics
                        my_metrics.update(loss, None, labels, None)

                    bar.next()  

            # Calculate metrics for the epoch
            results = my_metrics.calculate()
    
            if phase == 'train':
                train_metrics = {'loss': [results['loss']], 
                        }
            

            print('{} Loss: {:.4f}'.format(phase, results['loss']))

           # Save model and update best weights only if recall has improved
            if phase == 'val' and results['loss'] < best_loss:
                best_loss = results['loss']
                # best_model_wts = copy.deepcopy(model.state_dict())  
                model_out = model

                best_train_metrics = train_metrics
                best_val_metrics = {'loss': [results['loss']]
                                    }
    
                PATH = os.path.join(args.root, 'models', 'FAIGB_SAE_' + run_name)
  
                os.makedirs(os.path.join(args.root, 'models'), exist_ok=True)
                if args.save == 'model':
                    print('Saving model to: ' + PATH + '.pth')
                    try:
                        torch.save(model_out.module, PATH + '.pth')
                    except:
                         torch.save(model_out, PATH + '.pth')
                elif args.save == 'weights':
                    print('Saving model weights to: ' + PATH + '_weights.pth')
                    try:
                        torch.save(model_out.module.state_dict(), PATH + '.pth')
                    except:
                        torch.save(model_out.state_dict(), PATH + '.pth')
                elif args.save == 'both':
                    if args.arch != 'parallel':
                        print('Saving model and weights to: ' + PATH + '.pth and ' + PATH + '_weights.pth')
                        try:
                            torch.save(model_out.module, PATH + '.pth') 
                            torch.save(model_out.module.state_dict(), PATH + '_weights.pth')
                        except:
                            torch.save(model_out, PATH + '.pth')
                            torch.save(model_out.state_dict(), PATH + '_weights.pth')
                    
  
            if phase == 'val':
                val_loss_history.append(results['loss'])
            
            if phase == 'train':
                wandb.log({"Train_loss": results['loss']})  
            else:
                wandb.log({"Val_loss": results['loss']})
        
            # Reset metrics for the next epoch
            my_metrics.reset()

        
        bar.finish()
        epoch += 1
    
    wandb.finish()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Loss of saved model: {:4f}'.format(best_loss))
    return model_out, None, best_loss, None, run_name, best_train_metrics, best_val_metrics
