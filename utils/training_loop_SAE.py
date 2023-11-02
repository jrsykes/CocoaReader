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
import umap
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import pandas as pd

def train_model(args, model, optimizer, device, dataloaders_dict, criterion, patience, batch_size, num_classes, distances):      
    # @torch.compile
    # def run_model(x):
    #     return model(x)
        
    # Check environmental variable WANDB_MODE
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
    
    my_metrics = toolbox.Metrics(metric_names= ['cont_loss', 'MSE_loss'], num_classes=num_classes)


    reducer = umap.UMAP(n_components=2)
    if os.path.exists(os.path.join(args.root, 'umap_data.csv')):
        os.remove(os.path.join(args.root, 'umap_data.csv'))
    
    #sample images for visualisation
    len_ = len(dataloaders_dict['val'].dataset)
    selected_indices = [0, len_//10, len_//9, len_//8, len_//7, len_//6, len_//5, len_//4, len_//3]
    sampler = toolbox.NineImageSampler(selected_indices)
    sample_data_loader = DataLoader(dataloaders_dict['val'].dataset, batch_size=9, sampler=sampler)
    sample_images, _ = next(iter(sample_data_loader))
    sample_images = F.interpolate(sample_images, size=(356, 356), mode='bilinear', align_corners=True).to(device)

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

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

           #Get size of whole dataset split
            n = len(dataloaders_dict[phase].dataset)
           #Begin training
            print(phase)
            all_encoded = []
            all_labels = []
            mean_loss = 0.00
            with Bar('Learning...', max=n/batch_size+1) as bar:
               
                for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
                    
                    #Load images and lables from current batch onto GPU(s)
                    SRinputs = inputs.to(device)
                    inputs = F.interpolate(inputs, size=(356, 356), mode='bilinear', align_corners=True)
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        #Forward pass   
                        encoded, decoded = model(inputs)
                        
                        #Calculate losses and gradients then normalise the gradients
                        contrastive_loss = toolbox.contrastive_loss_with_dynamic_margin(encoded, distances, labels)
                        MSE_loss = criterion(decoded, SRinputs)          
                        l1_norm = torch.tensor(sum(p.abs().sum() for p in model.parameters() if p.dim() > 1).item(), requires_grad=True)
                        mean_loss += ((contrastive_loss + MSE_loss + l1_norm)/3).detach()
                                               
                        if phase == 'train':
                            optimizer.zero_grad()
                            total_loss = contrastive_loss + MSE_loss + l1_norm * args.l1_lambda
                            # total_loss = contrastive_loss + l1_norm * args.l1_lambda

                            total_loss.backward(retain_graph=True)
                            optimizer.step()

                        if phase == 'val':
                            all_encoded.append(encoded.cpu().detach().numpy())
                            all_labels.append(labels.cpu().detach().numpy())
                            if idx == 0:                                
                                _, sample_decoded = model(sample_images)
                            
                                grid = vutils.make_grid(sample_decoded, nrow=3, padding=0, normalize=False)
                                PATH = os.path.join(args.root, "reconstructions_" + args.model_name)
                                os.makedirs(PATH, exist_ok=True)
                                vutils.save_image(grid, os.path.join(PATH, "epoch_" + str(epoch) + ".png"))                      
                           
                        #Update metrics
                        my_metrics.update(cont_loss=contrastive_loss, MSE_loss=MSE_loss, labels=labels)

                    bar.next()  

            # Calculate metrics for the epoch
            results = my_metrics.calculate()
            

            if phase == 'train':
                train_metrics = {'cont_loss': results['cont_loss'], 
                                    'MSE_loss': results['MSE_loss']                                    
                                    }
            
            print('{} Contrastive loss: {:.4f} SR loss: {:.4f} '.format(phase, results['cont_loss'], results['MSE_loss']))

           # Save model and update best weights only if recall has improved
            if phase == 'val' and mean_loss < best_loss:
                best_loss = mean_loss
                val_loss_history.append(mean_loss)

                best_train_metrics = train_metrics
                best_val_metrics = {'cont_loss': results['cont_loss'], 
                                    'MSE_loss': results['MSE_loss']                                    
                                    }
    
                PATH = os.path.join(args.root, 'models', 'FAIGB_SAE_' + run_name)
  
                os.makedirs(os.path.join(args.root, 'models'), exist_ok=True)
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
                    if args.arch != 'parallel':
                        print('Saving model and weights to: ' + PATH + '.pth and ' + PATH + '_weights.pth')
                        try:
                            torch.save(model.module, PATH + '.pth') 
                            torch.save(model.module.state_dict(), PATH + '_weights.pth')
                        except:
                            torch.save(model, PATH + '.pth')
                            torch.save(model.state_dict(), PATH + '_weights.pth')
                     
            
            if phase == 'train':
                wandb.log({"Train_cont_loss": results['cont_loss'], "Train_MSE_loss": results['MSE_loss']})  
            else:
                wandb.log({"Val_cont_loss": results['cont_loss'], "Val_MSE_loss": results['MSE_loss']})
                                
                all_encoded_np = np.concatenate(all_encoded, axis=0)
                all_labels_np = np.concatenate(all_labels, axis=0)
     
                data_umap = reducer.fit_transform(all_encoded_np)
      
                UMAP_table = wandb.Table(columns=["UMAP_X", "UMAP_Y", "Label", "Epoch"])
                csv_data = []  # List to hold data for CSV

                for i in range(data_umap.shape[0]):
                    UMAP_table.add_data(data_umap[i, 0], data_umap[i, 1], all_labels_np[i], epoch)
                    csv_data.append([data_umap[i, 0], data_umap[i, 1], all_labels_np[i], epoch])

                # Convert the list of data to a pandas DataFrame
                df = pd.DataFrame(csv_data, columns=["UMAP_X", "UMAP_Y", "Label", "Epoch"])

                # Save or append the DataFrame to a CSV file
                csv_filename = os.path.join(args.root, 'umap_data.csv')
                
                with open(csv_filename, 'a') as f:
                    # If the file does not exist, write the header, otherwise append without the header
                    df.to_csv(f, header=f.tell()==0, index=False)

                # Log the table to wandb
                wandb.log({"UMAP_table": UMAP_table})

                # Clear the lists for the next epoch
                all_encoded.clear()
                all_labels.clear()

            # Reset metrics for the next epoch
            my_metrics.reset()

        
        bar.finish()
        epoch += 1
    
    wandb.finish()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Loss of saved model: {:4f}'.format(best_loss))
    return None, best_loss, None, run_name, best_train_metrics, best_val_metrics
