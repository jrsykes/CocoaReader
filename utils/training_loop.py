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
#import toolbox


def train_model(args, model, optimizer, device, dataloaders_dict, criterion, patience, initial_bias, input_size, batch_size, n_tokens=None, AttNet=None, ANoptimizer=None):      
    # @torch.compile
    # def run_model(x):
    #     return model(x)
    
    #Check environmental variable WANDB_MODE
    if args.WANDB_MODE == 'offline':   
        if args.sweep is False:
            if args.run_name is None:
                run_name = RandomWords().get_random_word() + '_' + str(time.time())[-2:]
                wandb.init(project=args.project_name, name=run_name, mode="offline")
            else:
                wandb.init(project=args.project_name, name=args.run_name, mode="offline")
                run_name = args.run_name
        else:
            run_name = wandb.run.name
    else:
        if args.sweep is False:
            if args.run_name is None:
                run_name = RandomWords().get_random_word() + '_' + str(time.time())[-2:]
                wandb.init(project=args.project_name, name=run_name)
            else:
                wandb.init(project=args.project_name, name=args.run_name)
                run_name = args.run_name
        else:
            run_name = wandb.run.name
    
    my_metrics = toolbox.Metrics()

    since = time.time()
    val_loss_history = []
    best_f1 = 0.0
    best_f1_acc = 0.0
    best_train_f1 = 0.0
    
    #Save current weights as 'best_model_wts' variable. 
    #This will be reviewed each epoch and updated with each improvment in validation recall
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_wts['module.fc.bias'] = initial_bias.to(device)

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
            if phase == 'train':
                model.train()  # Set model to training mode
                if AttNet != None:
                    AttNet.train()
 
            elif phase == 'val':
                model.eval()   # Set model to evaluate mode
                if AttNet != None:
                    AttNet.eval()


           #Get size of whole dataset split
            n = len(dataloaders_dict[phase].dataset)
           #Begin training
            print(phase)
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
                        outputs, outputs2 = model(inputs)

                        # #Reduce dimensionality of output to 32x2
                        # if args.split_image == True:
                        #     #old_output_shell = outputs[0:batch_size,0:num_classes]
                        #     new_batch = []
                        #     for i in range(batch_size):
                        #         outputs_ = outputs[i*n_tokens**2:(i+1)*n_tokens**2]
                        #         outputs_ = outputs_.detach().cpu()
                                
                        #         outputs_flat = torch.empty(0)
                        #         for i in range(outputs_.shape[1]):
                        #             outputs_flat = torch.cat((outputs_flat,outputs_[:,i]),0)

                        #         outputs_ = outputs_flat.to(device).unsqueeze(0)
                        #         outputs_ = AttNet(outputs_)
                                                        
                        #         new_batch.append(outputs_)

                        #     outputs = torch.stack(new_batch, dim=0).squeeze(1)

                        # if phase == 'train':
                        #    loss, step = criterion[phase](outputs, labels, step)
                        # else:
                        #    loss = criterion[phase](outputs, labels)
                        
                        
                        # define empty torch vector
                        labels2 = []
                        for i in (labels.tolist()):
                            if i == 1:
                                labels2.append(1)
                            else:
                                labels2.append(0)
                        labels2 = torch.tensor(labels2).to(device)
                              
                        loss = criterion(outputs, labels)
                        loss += criterion(outputs2, labels2)
  
                        l1_norm = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
                        loss += args.l1_lambda * l1_norm
                        
                        _, preds = torch.max(outputs, 1)    
                        stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), digits=4, output_dict = True, zero_division = 0)
                        stats_out = stats['weighted avg']
                       
                  
                       # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()  
                            if ANoptimizer != None:
                                ANoptimizer.step() 
      

                        #Update metrics
                        my_metrics.update(loss, preds, labels, stats_out)

                    bar.next()  

            # Calculate metrics for the epoch
            epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1 = my_metrics.calculate()
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            if phase == 'train':
                train_f1 = epoch_f1 
            

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('{} Precision: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(phase, epoch_precision, epoch_recall, epoch_f1))
            # Save statistics to tensorboard log
            

           # Save model and update best weights only if recall has improved
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_f1_acc = epoch_acc
                best_f1_loss = epoch_loss
                best_train_f1 = train_f1
                best_model_wts = copy.deepcopy(model.state_dict())  
                model_out = model
    
                PATH = os.path.join(args.root, 'dat/models', args.model_name)
                if args.save == 'model':
                    print('Saving model to: ' + PATH + '.pth')
                    try:
                        torch.save(model.module, PATH + '.pth')
                    except:
                         torch.save(model, PATH + '.pth')
                elif args.save == 'weights':
                    print('Saving model weights to: ' + PATH + '_weights.pth')
                    try:
                        torch.save(model.module.state_dict(), PATH + '.pth')
                    except:
                        torch.save(model.state_dict(), PATH + '.pth')
                elif args.save == 'both':
                    print('Saving model and weights to: ' + PATH + '.pth and ' + PATH + '_weights.pth')
                    try:
                        torch.save(model.module, PATH + '.pth') 
                        torch.save(model.module.state_dict(), PATH + '_weights.pth')
                    except:
                        torch.save(model, PATH + '.pth')
                        torch.save(model.state_dict(), PATH + '_weights.pth')
  
            if phase == 'val':
                val_loss_history.append(epoch_loss)
            
            if phase == 'train':
                wandb.log({"epoch": epoch, "Train_loss": epoch_loss, "Train_acc": epoch_acc, "Train_F1": epoch_f1, "Best_train_f1": best_train_f1})  
            else:
                wandb.log({"epoch": epoch, "Val_loss": epoch_loss, "Val_acc": epoch_acc, "Val_F1": epoch_f1, "Best_F1": best_f1, "Best_F1_acc": best_f1_acc})
        
            # Reset metrics for the next epoch
            my_metrics.reset()

        
        bar.finish() 
        epoch += 1
    input_size = inputs.size()[1:]
    
    GFLOPs, params = toolbox.count_flops(model=model, device=device, input_size=input_size)
    wandb.log({'GFLOPs': GFLOPs, 'params': params})
    
    wandb.finish()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Acc of saved model: {:4f}'.format(best_f1_acc))
    print('F1 of saved model: {:4f}'.format(best_f1))
    return model_out, best_f1, best_f1_loss, best_train_f1, run_name
