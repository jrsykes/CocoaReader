from __future__ import print_function
from __future__ import division


import torch
import numpy as np
import time
import wandb
from progress.bar import Bar
import os
from random_word import RandomWords
import toolbox
import umap
import pandas as pd
import RobinsonFoulds

import networkx as nx
from ete3 import Tree
from phylo2vec.base import to_vector, to_newick



def train_model(args, model, optimizer, device, dataloaders_dict, criterion, patience, batch_size, num_classes, taxonomy, ESS_alpha, MSE_alpha, scheduler):      
   
    MSE_criterion = torch.nn.MSELoss(reduction='mean')
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
    
    my_metrics = toolbox.Metrics(metric_names= ['loss', 'ESS', 'MSE', 'L1'], num_classes=num_classes)

    reducer = umap.UMAP(n_components=3)
    if os.path.exists(os.path.join(args.root, 'umap_data.csv')):
        os.remove(os.path.join(args.root, 'umap_data.csv'))
    

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
                torch.no_grad()
                model.eval()   # Set model to evaluate mode

           #Get size of whole dataset split
            n = len(dataloaders_dict[phase].dataset)
           #Begin training
            print(phase)
            all_encoded = []
            all_labels = []
            epoch_loss = 0.00
            with Bar('Learning...', max=n/batch_size) as bar:
               
                for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):

                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    

                    with torch.set_grad_enabled(phase == 'train'):
                        #Forward pass   
                        encoded_pooled, _ = model(inputs)
                        # encoded_pooled = model(inputs)
        
                        trees, name_to_base_name = RobinsonFoulds.trees(taxonomy, labels, encoded_pooled)
       
                        matrices = RobinsonFoulds.generate_matrices(trees, name_to_base_name)

                        MSE = MSE_criterion(matrices['pred_matrix'], matrices['target_matrix']) * MSE_alpha
                  
                        # #Edge similarity score
                        ESS = RobinsonFoulds.ESS(trees["target_tree"], trees["pred_tree"]) * ESS_alpha
                   
                        #output tree as graphical representation
                        trees["target_tree"].write(format=1, outfile="/users/jrs596/tree_target.newick")
                        trees["pred_tree"].write(format=1, outfile="/users/jrs596/tree_pred.newick")

                        l1_norm = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1) * args.l1_lambda

                        loss = MSE  + ESS # + l1_norm   
                        # loss = MSE + ESS  

                        epoch_loss += loss

                        if phase == 'train':
                            optimizer.zero_grad()

                            loss.backward()
                            optimizer.step()

                        if phase == 'val':
                            all_encoded.append(encoded_pooled.cpu().detach().numpy())
                            all_labels.append(labels.cpu().detach().numpy())
                            if idx == 0:                                

                                encoded_pooled, _ = model(inputs)
                                # encoded_pooled = model(inputs)
                                trees, _ = RobinsonFoulds.trees(taxonomy, labels, encoded_pooled)

                                PATH = os.path.join(args.root, "trees_" + args.model_name)
                                os.makedirs(PATH, exist_ok=True)
                                trees["pred_tree"].write(format=1, outfile=os.path.join(PATH, str(epoch) + "tree_pred.newick"))
                                trees["target_tree"].write(format=1, outfile=os.path.join(PATH, str(epoch) + "tree_target.newick"))                

                        #Update metrics
                        my_metrics.update(loss=loss ,ESS=ESS, MSE=MSE, L1=l1_norm, labels=labels)
                    
                    bar.next()  

            # Calculate metrics for the epoch
            results = my_metrics.calculate()
            

            if phase == 'train':
                train_metrics = {'loss': results['loss'], 'ESS': results['ESS'], 'MSE': results['MSE'], 'L1': results['L1']}    
            else:
                scheduler.step(epoch_loss)
                                                            
            print('{} loss: {:.4f} ESS: {:.4f} MSE: {:.4f} L1_norm: {:.4f}'.format(phase, results['loss'], results['ESS'], results['MSE'], results['L1']))                

           # Save model and update best weights only if recall has improved
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                val_loss_history.append(epoch_loss)

                best_train_metrics = train_metrics
                best_val_metrics = {'loss' : results['loss'], 'ESS': results['ESS'], 'MSE': results['MSE'], 'L1': results['L1']}  
    
                PATH = os.path.join(args.root, 'models', args.model_name)
  
                os.makedirs(os.path.join(args.root, 'models'), exist_ok=True)

                if args.save:
                    print('Saving model weights to: ' + PATH + '_weights.pth')
                    try:
                        torch.save(model.module.state_dict(), PATH + '.pth')
                    except:
                        torch.save(model.state_dict(), PATH + '.pth')
                     
            
            if phase == 'train':
                wandb.log({"Train_loss": results['loss'], "Train_ESS": results['ESS'], "Train_MSE": results['MSE'], "Train_L1_norm": results['L1']})
            else:
                wandb.log({"Val_loss": results['loss'], "Val_ESS": results['ESS'], "Val_MSE": results['MSE'], "Val_L1_norm": results['L1']})
                                                
                all_encoded_np = np.concatenate(all_encoded, axis=0)
                all_labels_np = np.concatenate(all_labels, axis=0)
     
                data_umap = reducer.fit_transform(all_encoded_np)
           
                # UMAP_table = wandb.Table(columns=["UMAP_X", "UMAP_Y", "UMAP_Z", "Label", "Epoch"])
                csv_data = []  # List to hold data for CSV

                for i in range(data_umap.shape[0]):
                    # UMAP_table.add_data(data_umap[i, 0], data_umap[i, 1], data_umap[i, 2], all_labels_np[i], epoch)
                    csv_data.append([data_umap[i, 0], data_umap[i, 1], data_umap[i, 2], all_labels_np[i], epoch])
            
                # Convert the list of data to a pandas DataFrame
                df = pd.DataFrame(csv_data, columns=["UMAP_X", "UMAP_Y", "UMAP_Z", "Label", "Epoch"])

                with open(UMAP_csv_filename, 'a') as f:
                    # If the file does not exist, write the header, otherwise append without the header
                    df.to_csv(f, header=f.tell()==0, index=False)

                # Log the table to wandb
                # wandb.log({"UMAP_table": UMAP_table})

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
