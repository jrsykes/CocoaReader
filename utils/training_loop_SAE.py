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
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from Bio import Phylo
from collections import defaultdict

def train_model(args, model, optimizer, device, dataloaders_dict, criterion, patience, batch_size, num_classes, distances):      
   
    
    # Save or append the DataFrame to a CSV file
    UMAP_csv_filename = os.path.join(args.root, args.model_name + '_umap_data.csv')
    if os.path.exists(UMAP_csv_filename):
    # Delete the file
        os.remove(UMAP_csv_filename)
        
    # Check environmental variable WANDB_MODE
    run_name = None
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
    
    my_metrics = toolbox.Metrics(metric_names= ['Genetic_loss', 'SR_loss', 'Euclid_loss'], num_classes=num_classes)

    constructor = DistanceTreeConstructor()
    reducer = umap.UMAP(n_components=3)
    if os.path.exists(os.path.join(args.root, 'umap_data.csv')):
        os.remove(os.path.join(args.root, 'umap_data.csv'))
    
    #sample images for visualisation
    len_ = len(dataloaders_dict['val'].dataset)
    selected_indices = [0, len_//10, len_//9, len_//8, len_//7, len_//6, len_//5, len_//4, len_//3]
    selected_indices = [i for i in range(args.batch_size)]
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
            total_loss = 0.00
            with Bar('Learning...', max=n/batch_size) as bar:
               
                for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):

                    rows, cols = labels.numpy().tolist(), labels.numpy().tolist()
                    labels_dist = distances.iloc[rows, cols]
                    label_relationship_matrix = torch.tensor(labels_dist.values, dtype=torch.float).to(device)
             
                    #Load images and lables from current batch onto GPU(s)
                    SRinputs = inputs.to(device)
                    inputs = F.interpolate(inputs, size=(356, 356), mode='bilinear', align_corners=True)
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        #Forward pass   
                        encoded, SRdecoded, predicted_distances, poly_euclid_distances = model(inputs)

                        # euclid_distances = torch.cdist(encoded, encoded, p=2)

                        #Calculate losses and gradients then normalise the gradients
                        # contrastive_loss = toolbox.contrastive_loss_with_dynamic_margin(encoded, distances, labels)/100    # weight to put on sensible scale
                        SR_loss = criterion(SRdecoded, SRinputs)/100000                                                      # weight to put on sensible scale    
                        genetic_loss = criterion(predicted_distances, torch.log(label_relationship_matrix+1e-6)) / 10000
                        euclid_loss = criterion(poly_euclid_distances, torch.log(label_relationship_matrix+1e-6)) / 10000
                        
                        # euclid_loss = criterion(poly_euclid_distances, label_relationship_matrix) / 10000
                        # euclid_loss, _ = toolbox.contrastive_loss_with_dynamic_margin(encoded, distances, labels)
                        # euclid_loss = euclid_loss / 100
                        
                        l1_norm = torch.tensor(sum(p.abs().sum() for p in model.parameters() if p.dim() > 1).item(), requires_grad=True)
                        total_loss += (genetic_loss + SR_loss + euclid_loss).detach()
                                               

                        if phase == 'train':
                            optimizer.zero_grad()
                            quad_loss = genetic_loss + SR_loss + euclid_loss + l1_norm * args.l1_lambda

                            quad_loss.backward(retain_graph=True)
                            optimizer.step()

                        if phase == 'val':
                            all_encoded.append(encoded.cpu().detach().numpy())
                            all_labels.append(labels.cpu().detach().numpy())
                            if idx == 0:                                
                                _, SRdecoded, _, _ = model(sample_images)
                            
                                grid = vutils.make_grid(SRdecoded, nrow=3, padding=0, normalize=False)
                                PATH = os.path.join(args.root, "reconstructions_" + args.model_name)
                                os.makedirs(PATH, exist_ok=True)
                                vutils.save_image(grid, os.path.join(PATH, "epoch_" + str(epoch) + ".png"))     
                                ################################################################################################
                                names = labels_dist.columns.tolist()

                                name_count = defaultdict(int)
                                unique_names = []
                                for name in names:
                                    if name_count[name]:
                                        unique_name = f"{name}_{name_count[name]}"
                                    else:
                                        unique_name = name
                                    name_count[name] += 1
                                    unique_names.append(unique_name)


                                lower_predicted_distances = toolbox.lower_triangle(predicted_distances.cpu().detach().numpy())
                                lower_label_relationship_matrix = toolbox.lower_triangle(label_relationship_matrix.cpu().detach().numpy())

                                tree_pred = constructor.upgma(_DistanceMatrix(names=unique_names, matrix=lower_predicted_distances))            
                                tree_true = constructor.upgma(_DistanceMatrix(names=unique_names, matrix=lower_label_relationship_matrix))                     
                                PATH = os.path.join(args.root, "trees_" + args.model_name)
                                os.makedirs(PATH, exist_ok=True)
                                Phylo.write(tree_pred, os.path.join(PATH, str(epoch) + "tree_pred.newick"), "newick") 
                                Phylo.write(tree_true, os.path.join(PATH, str(epoch) + "tree_true.newick"), "newick")                   
                           
                        #Update metrics
                        my_metrics.update(Genetic_loss=genetic_loss, SR_loss=SR_loss, Euclid_loss=euclid_loss, labels=labels)
                    
  
                    bar.next()  

            # Calculate metrics for the epoch
            results = my_metrics.calculate()
            

            if phase == 'train':
                train_metrics = {'Genetic_loss': results['Genetic_loss'], 
                                    'SR_loss': results['SR_loss'], 'Euclid_loss': results['Euclid_loss']                                    
                                    }
            
            print('{} Genetic loss: {:.4f} SR loss: {:.4f} Euclid loss: {:.4f} '.format(phase, results['Genetic_loss'], results['SR_loss'], results['Euclid_loss']))
            

           # Save model and update best weights only if recall has improved
            if phase == 'val' and total_loss < best_loss:
                best_loss = total_loss
                val_loss_history.append(total_loss)

                best_train_metrics = train_metrics
                best_val_metrics = {'Genetic_loss': results['Genetic_loss'], 
                                    'SR_loss': results['SR_loss'], 'Euclid_loss': results['Euclid_loss']                                    
                                    }
    
                PATH = os.path.join(args.root, 'models', args.model_name + '_epoch_' + str(epoch))
  
                os.makedirs(os.path.join(args.root, 'models'), exist_ok=True)
                # if args.save == 'model':
                #     print('Saving model to: ' + PATH + '.pth')
                #     try:
                #         torch.save(model.module, PATH + '.pth')
                #     except:
                #          torch.save(model, PATH + '.pth')
                # elif args.save == 'weights':
                if args.save:
                    print('Saving model weights to: ' + PATH + '_weights.pth')
                    try:
                        torch.save(model.module.state_dict(), PATH + '.pth')
                    except:
                        torch.save(model.state_dict(), PATH + '.pth')
                # elif args.save == 'both':
                #     if args.arch != 'parallel':
                #         print('Saving model and weights to: ' + PATH + '.pth and ' + PATH + '_weights.pth')
                #         try:
                #             torch.save(model.module, PATH + '.pth') 
                #             torch.save(model.module.state_dict(), PATH + '_weights.pth')
                #         except:
                #             torch.save(model, PATH + '.pth')
                #             torch.save(model.state_dict(), PATH + '_weights.pth')
                     
            
            if phase == 'train':
                wandb.log({"Train_Genetic_loss": results['Genetic_loss'], "Train_SR_loss": results['SR_loss'], "Train_Euclid_loss": results['Euclid_loss']})  
            else:
                wandb.log({"Val_Genetic_loss": results['Genetic_loss'], "Val_SR_loss": results['SR_loss'], "Val_Euclid_loss": results['Euclid_loss']})
                                
                all_encoded_np = np.concatenate(all_encoded, axis=0)
                all_labels_np = np.concatenate(all_labels, axis=0)
     
                data_umap = reducer.fit_transform(all_encoded_np)
           
                UMAP_table = wandb.Table(columns=["UMAP_X", "UMAP_Y", "UMAP_Z", "Label", "Epoch"])
                csv_data = []  # List to hold data for CSV

                for i in range(data_umap.shape[0]):
                    UMAP_table.add_data(data_umap[i, 0], data_umap[i, 1], data_umap[i, 2], all_labels_np[i], epoch)
                    csv_data.append([data_umap[i, 0], data_umap[i, 1], data_umap[i, 2], all_labels_np[i], epoch])
            
                # Convert the list of data to a pandas DataFrame
                df = pd.DataFrame(csv_data, columns=["UMAP_X", "UMAP_Y", "UMAP_Z", "Label", "Epoch"])

                with open(UMAP_csv_filename, 'a') as f:
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
