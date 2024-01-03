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
# import umap
import torchvision.utils as vutils
# from torch.utils.data import DataLoader
import pandas as pd
from torchvision import datasets, transforms
import os
from PIL import Image

# from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
# from Bio import Phylo
# from collections import defaultdict
# import RobinsonFoulds

def train_model(args, model, optimizer, device, dataloaders_dict, criterion, patience, batch_size, num_classes, taxonomy):      
   
	
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
	
	my_metrics = toolbox.Metrics(metric_names= ['SR_loss', 'L1'], num_classes=num_classes)

	# constructor = DistanceTreeConstructor()
	# reducer = umap.UMAP(n_components=3)
	# if os.path.exists(os.path.join(args.root, 'umap_data.csv')):
	#     os.remove(os.path.join(args.root, 'umap_data.csv'))
	
	# #sample images for visualisation
	# len_ = len(dataloaders_dict['val'].dataset)
	# selected_indices = [0, len_//10, len_//9, len_//8, len_//7, len_//6, len_//5, len_//4, len_//3]
	# selected_indices = [i for i in range(args.batch_size)]
	# sampler = toolbox.NineImageSampler(selected_indices)
	# sample_data_loader = DataLoader(dataloaders_dict['val'].dataset, batch_size=9, sampler=sampler)
	# sample_images, _ = next(iter(sample_data_loader))
	# sample_images = F.interpolate(sample_images, size=(356, 356), mode='bilinear', align_corners=True).to(device)
	# Define the directory where the images are located
	data_dir = '/users/jrs596/scratch/dat/sample_SR_images'

	# Define the transformation to apply to the images
	transform = transforms.Compose([
		transforms.ToTensor()  # Convert the images to tensors
	])

	# Create the dataset
	dataset = datasets.ImageFolder(data_dir, transform=transform)
	sample_dataloader = torch.utils.data.DataLoader(dataset, batch_size=9, shuffle=False)
	
	
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

					#Load images and lables from current batch onto GPU(s)
					SRinputs = inputs.to(device)
					
					#PhyNet
					inputs = F.interpolate(inputs, size=(356, 356), mode='bilinear', align_corners=True)                   
					#ResNet18
					# inputs = F.interpolate(inputs, size=(375, 375), mode='bilinear', align_corners=True)
					
					inputs = inputs.to(device)
					labels = labels.to(device)
					
					with torch.set_grad_enabled(phase == 'train'):
						#Forward pass   
						_, SRdecoded = model(inputs)
				   
						#Calculate losses and gradients then normalise the gradients
						SR_loss = criterion(SRdecoded, SRinputs)/100000                                                      # weight to put on sensible scale    


						l1_norm = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1) 
						loss = SR_loss + l1_norm * args.l1_lambda                 
						epoch_loss += loss
      
						if phase == 'train':
							optimizer.zero_grad()

							loss.backward()
							optimizer.step()

						if phase == 'val':
					
							PATH = os.path.join(args.root, "reconstructions_" + args.model_name, "epoch_" + str(epoch))
							os.makedirs(PATH, exist_ok=True)
							
							if idx == 0:
								for image, _ in sample_dataloader:
									print("\nSaving sample images")    
									_, SRdecoded = model(image.to(device))
						
								for idx, img in enumerate(SRdecoded):
									#Crop one picel from right and bottom of image
									img = img[:, :-10, :-10]
									img = img.squeeze(0)
									image = img * 255.0
									image = torch.clamp(image, 0, 255)
									image = image.byte()
									# Convert to PIL and save
									img_pil = transforms.ToPILImage()(image.cpu().detach())
									img_pil.save(os.path.join(PATH, f"image_{idx}.jpeg"))             

						#Update metrics
						my_metrics.update(SR_loss=SR_loss, L1=l1_norm, labels=labels)

					bar.next()  

			# Calculate metrics for the epoch
			results = my_metrics.calculate()
			

			if phase == 'train':
				train_metrics = {'SR_loss': results['SR_loss'], 'L1': results['L1']}                              
									
			print('{} SR_loss: {:.4f} L1_norm: {:.4f}'.format(phase, results['SR_loss'], results['L1']))                

		   # Save model and update best weights only if recall has improved
			if phase == 'val' and epoch_loss < best_loss:
				best_loss = epoch_loss
				val_loss_history.append(epoch_loss)

				best_train_metrics = train_metrics
				best_val_metrics = {'SR_loss': results['SR_loss'], 'L1': results['L1']}  
	
				PATH = os.path.join(args.root, 'models', args.model_name)
  
				os.makedirs(os.path.join(args.root, 'models'), exist_ok=True)

				if args.save:
					print('Saving model weights to: ' + PATH + '_weights.pth')
					try:
						torch.save(model.module.state_dict(), PATH + '.pth')
					except:
						torch.save(model.state_dict(), PATH + '.pth')
					 
			
			if phase == 'train':
				wandb.log({"Train_SR_loss": results['SR_loss'], "Train_L1_norm": results['L1']})
			else:
				wandb.log({"Val_SR_loss": results['SR_loss'], "Val_L1_norm": results['L1']})
												
				# all_encoded_np = np.concatenate(all_encoded, axis=0)
				# all_labels_np = np.concatenate(all_labels, axis=0)
	 
				# data_umap = reducer.fit_transform(all_encoded_np)
		   
				# UMAP_table = wandb.Table(columns=["UMAP_X", "UMAP_Y", "UMAP_Z", "Label", "Epoch"])
				# csv_data = []  # List to hold data for CSV

				# for i in range(data_umap.shape[0]):
				#     # UMAP_table.add_data(data_umap[i, 0], data_umap[i, 1], data_umap[i, 2], all_labels_np[i], epoch)
				#     csv_data.append([data_umap[i, 0], data_umap[i, 1], data_umap[i, 2], all_labels_np[i], epoch])
			
				# # Convert the list of data to a pandas DataFrame
				# df = pd.DataFrame(csv_data, columns=["UMAP_X", "UMAP_Y", "UMAP_Z", "Label", "Epoch"])

				# with open(UMAP_csv_filename, 'a') as f:
				#     # If the file does not exist, write the header, otherwise append without the header
				#     df.to_csv(f, header=f.tell()==0, index=False)

				# Log the table to wandb
				# wandb.log({"UMAP_table": UMAP_table})

				# # Clear the lists for the next epoch
				# all_encoded.clear()
				# all_labels.clear()

			# Reset metrics for the next epoch
			my_metrics.reset()

		
		bar.finish()
		epoch += 1
	
	wandb.finish()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Loss of saved model: {:4f}'.format(best_loss))
	return None, best_loss, None, run_name, best_train_metrics, best_val_metrics