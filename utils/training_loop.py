from __future__ import print_function
from __future__ import division


import torch
import numpy as np
import time
import copy
import wandb
from sklearn import metrics
from progress.bar import Bar
import os
#import sys
from random_word import RandomWords
import toolbox
#sys.path.append(os.path.join(os.getcwd(), 'scripts/CocoaReader/utils'))
from toolbox import DynamicFocalLoss

def train_model(args, model, optimizer, device, dataloaders_dict, criterion, patience, batch_size, num_classes, best_f1):      
    # Check environmental variable WANDB_MODE
    # if args.wandb_MODE == 'offline':   
    #     if args.sweep_config == None:
    #         if args.run_name is None:
    #             run_name = RandomWords().get_random_word() + '_' + str(time.time())[-2:]
    #             wandb.init(project=args.project_name, name=run_name, mode="offline")
    #         else:
    #             wandb.init(project=args.project_name, name=args.run_name, mode="offline")
    #             run_name = args.run_name
    #     else:
    #         run_name = wandb.run.name
    # else:
        # if args.sweep_config == None:
        #     if args.run_name is None:
        #         # run_name = RandomWords().get_random_word() + '_' + str(time.time())[-2:]
        #         wandb.init(project=args.project_name, name=run_name)
        #     else:
        #         wandb.init(project=args.project_name, name=args.run_name)
        #         run_name = args.run_name
        # else:
        #     run_name = wandb.run.name

    my_metrics = toolbox.Metrics(metric_names='All', num_classes=num_classes)

    since = time.time()
    # val_loss_history = []
    val_F1_history = []
    # best_f1 = 0.0
    best_f1_acc = 0.0
    best_train_f1 = 0.0
    model_out = model
    #Save current weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_model_wts['module.fc.bias'] = initial_bias.to(device)

    epoch = 0
    # Run untill validation loss has not improved for n epochs=patience or max_epochs is reached
    while patience > 0 and epoch < args.max_epochs:
        print('\nEpoch {}'.format(epoch))
        print('-' * 10) 
        step  = 0
        #Ensure minimum number of epochs is met before patience is allow to reduce
        if len(val_F1_history) > args.min_epochs:
           #If the current loss is not at least 0.5% less than the lowest loss recorded, reduce patiece by one epoch
            if val_F1_history[-1] > min(val_F1_history)*args.beta:
                patience -= 1
            elif val_F1_history[-1] == np.nan:
                patience -= 1
            else:
               #If validation loss improves, reset patient to initial value
               patience = args.patience
        print('Patience: ' + str(patience) + '/' + str(args.patience))
   
       #Training and validation loop
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
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

                        outputs = model(inputs)


                        loss = criterion(outputs, labels)

                        l1_norm = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1) * args.l1_lambda
                        
                        loss += l1_norm                     
                        
                        _, preds = torch.max(outputs, 1)    
                        stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
                        stats_out = stats['weighted avg']
                       
                  
                       # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()  
                           
                        #Update metrics
                        my_metrics.update(loss=loss, preds=preds, labels=labels, stats_out=stats_out)

                    bar.next()  

            # Calculate metrics for the epoch
            epoch_metrics = my_metrics.calculate()
       
            if phase == 'train':
                train_f1 = epoch_metrics['f1'] 
                # train_metrics = {'loss': epoch_metrics['loss'], 
                #        'f1': epoch_metrics['f1'], 
                #        'acc': epoch_metrics['acc'].item(), 
                #        'precision': epoch_metrics['precision'], 
                #        'recall': epoch_metrics['recall'], 
                #        'BPR_F1': epoch_metrics['f1_per_class'][0], 'FPR_F1': epoch_metrics['f1_per_class'][1], 'Healthy_F1': epoch_metrics['f1_per_class'][2], 'WBD_F1': epoch_metrics['f1_per_class'][3]
                #         }
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_metrics['loss'], epoch_metrics['acc']))
            print('{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, epoch_metrics['precision'], epoch_metrics['recall'], epoch_metrics['f1']))           

           # Save model and update best weights only if recall has improved
            if phase == 'val' and epoch_metrics['f1'] > best_f1:
                best_f1 = epoch_metrics['f1']
                best_f1_acc = epoch_metrics['acc']
                # best_f1_loss = epoch_metrics['loss']
                best_train_f1 = train_f1
                # best_model_wts = copy.deepcopy(model.state_dict())  
                model_out = model

                # best_train_metrics = train_metrics
                # best_val_metrics = {'loss': epoch_metrics['loss'], 
                #                     'f1': epoch_metrics['f1'], 
                #                     'acc': epoch_metrics['acc'].item(), 
                #                     'precision': epoch_metrics['precision'], 
                #                     'recall': epoch_metrics['recall'], 
                #                     'BPR_F1': epoch_metrics['f1_per_class'][0], 'FPR_F1': epoch_metrics['f1_per_class'][1], 'Healthy_F1': epoch_metrics['f1_per_class'][2], 'WBD_F1': epoch_metrics['f1_per_class'][3]
                #                     }
    
                PATH = os.path.join(args.root, 'models', args.model_name)
                os.makedirs(os.path.join(args.root, 'models'), exist_ok=True)

                if args.save:
                    print('Saving model weights to: ' + PATH + '.pth')
                    try:
                        torch.save(model.module.state_dict(), PATH + '.pth')
                    except:
                        torch.save(model.state_dict(), PATH + '.pth')
                    
  
            if phase == 'val':
                val_F1_history.append(epoch_metrics['f1'])
            
            if phase == 'train':
                wandb.log({"Train_loss": epoch_metrics['loss'], "Train_acc": epoch_metrics['acc'], "Train_F1": epoch_metrics['f1'], "Best_train_f1": best_train_f1})  
            else:
                wandb.log({"Val_loss": epoch_metrics['loss'], "Val_acc": epoch_metrics['acc'], "Val_F1": epoch_metrics['f1'], "Best_F1": best_f1, "Best_F1_acc": best_f1_acc})
        
            # Reset metrics for the next epoch
            my_metrics.reset()

        
        bar.finish()
        epoch += 1
    

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Acc of saved model: {:4f}'.format(best_f1_acc))
    print('F1 of saved model: {:4f}'.format(best_f1))
    return model_out, best_f1
