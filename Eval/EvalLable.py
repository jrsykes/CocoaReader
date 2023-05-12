import torch
import os
import shutil
from torchvision import transforms, datasets



def eval(root, model, moved_count, input_size, Diff_Unsure):
	print()
	print('Evaluating model...')
	device = torch.device("cuda:0")
	model.eval()
	model = model.to(device)
	
	data_transforms = transforms.Compose([
            transforms.Resize((input_size,input_size)),
            transforms.ToTensor()])
    
	image_dataset = datasets.ImageFolder(os.path.join(root, Diff_Unsure), data_transforms)

	dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=True)

	for i, (inputs, labels) in enumerate(dataloader,0):
		inputs = inputs.to(device)
		labels = labels.to(device)
		
		outputs = model(inputs).mean(dim=0).unsqueeze(0)
		_, preds = torch.max(outputs, 1)
		
		if labels.item() == 0:
			class_ = 'BPR'
		elif labels.item() == 1:
			class_ = 'FPR'
		elif labels.item() == 2:
			class_ = 'Healthy'
		elif labels.item() == 3:
			class_ = 'WBD'
		
		file_name = dataloader.dataset.samples[i][0].split('/')[-1]
		#get model confidence for prediction
		confidence = torch.nn.functional.softmax(outputs, dim=1)[0][preds.item()].item()
		if Diff_Unsure == 'Difficult':
			threshold = 0.8
		elif Diff_Unsure == 'Unsure':
			threshold = 0.98
		if preds.item() == labels.item() and confidence > threshold:
			file_path = dataloader.dataset.samples[i][0]
			dest = root + os.path.join("/Easy/train", class_, file_name)

			shutil.move(file_path, dest)
			
			moved_count += 1
	print('\n Number of difficult or unsure images moved to train set: ', str(moved_count), '\n')
	return moved_count

