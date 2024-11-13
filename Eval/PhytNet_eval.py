import torch
from torchvision import models
import os
from torch import nn
from sklearn import metrics
import time
import sys
<<<<<<< HEAD
import toolbox

=======
>>>>>>> refs/remotes/origin/main

sys.path.append('/home/userfs/j/jrs596/scripts/CocoaReader/utils')

import toolbox



# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Unsure"
# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Difficult"

data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split_NotCooca/Easy"
# data_dir = "/users/jrs596/scratch/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split/Easy"


<<<<<<< HEAD
# data_dir = "/users/jrs596/scratch/dat/IR_split"
=======
data_dir = "/local/scratch/jrs596/dat/Ecuador/EcuadorWebImages_EasyDif_FinalClean_Compress500_split_NotCooca/Easy"
>>>>>>> refs/remotes/origin/main
num_classes = len(os.listdir(os.path.join(data_dir, 'val'))) 

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

	
<<<<<<< HEAD
config = {
        'beta1': 0.9305889820653824,  
        'beta2': 0.977926878163776,  
        'dim_1': 125,  
        'dim_2': 80,  
        'dim_3': 54, 
        'input_size': 308,  
        'kernel_1': 3,  
        'kernel_2': 11,  
        'kernel_3': 17,  
        'learning_rate': 0.0007653560770141792,  
        'num_blocks_1': 3,  
        'num_blocks_2': 2,  
        'out_channels': 5,  
        'num_heads': 3, #3 for PhytNetV0, 4 for ResNet18  
        'batch_size': 200,  
        'num_decoder_layers': 4,
    }

model = toolbox.build_model(num_classes=config['out_channels'], arch='PhytNetV0', config=config)

weights_path = "/users/jrs596/scratch/models/PhytNet-Cocoa-SemiSupervised_DFLoss-Discrim-PreTrained.pth"
# weights_path = "/users/jrs596/scratch/models/PhytNet67k-Cocoa-SemiSupervised_NotCocoa_OptDFLoss.pth"

# weights_path = '/users/jrs596/scratch/models/PhytNet-Cocoa-ablation.pth'

PhytNetWeights = torch.load(weights_path, map_location=device)      

model.load_state_dict(PhytNetWeights, strict=True)
input_size = config['input_size']
print('\nLoaded weights from: ', weights_path)
=======
# config = {
#         "beta1": 0.9051880132274126,
#         "beta2": 0.9630258300974864,
#         "dim_1": 49,
#         "dim_2": 97,
#         "dim_3": 68,
#         "kernel_1": 11,
#         "kernel_2": 9,
#         "kernel_3": 13,
#         "learning_rate": 0.0005921981578304907,
#         "num_blocks_1": 2,
#         "num_blocks_2": 4,
#         "out_channels": 7,
#         "input_size": 285,
#     }
 
# model = toolbox.build_model(num_classes=config['out_channels'], arch='PhytNetV0_ablation', config=config)

# # weights_path = "/users/jrs596/scratch/models/PhytNet183k-Cocoa-SemiSupervised_NotCocoa_DFLoss2.pth"
# # weights_path = "/users/jrs596/scratch/models/PhytNet67k-Cocoa-SemiSupervised_NotCocoa_OptDFLoss.pth"

# weights_path = '/users/jrs596/scratch/models/PhytNet-Cocoa-ablation.pth'

# PhyloNetWeights = torch.load(weights_path, map_location=device)


# model.load_state_dict(PhyloNetWeights, strict=True)
# input_size = config['input_size']
# print('\nLoaded weights from: ', weights_path)
>>>>>>> refs/remotes/origin/main

# resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-SemiSupervised_NotCocoa.pth"
# resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-IN-PT.pth"
resnet18_cococa_weights = "/users/jrs596/scratch/models/ResNet18-Cocoa-SemiSupervised_NotCocoa_DFLoss2.pth"

ResNet18Weights = torch.load(resnet18_cococa_weights, map_location=device)

model = models.resnet18(weights=None)
in_feat = model.fc.in_features
model.fc = nn.Linear(in_feat, 5)
model.load_state_dict(ResNet18Weights, strict=True)
input_size = 375

model.eval()   # Set model to evaluate mode
model = model.to(device)

<<<<<<< HEAD
batch_size = 200
=======
batch_size = 6
>>>>>>> refs/remotes/origin/main
criterion = nn.CrossEntropyLoss()

image_datasets = toolbox.build_datasets(data_dir=data_dir, input_size=input_size) #If images are pre compressed, use input_size=None, else use input_size=args.input_size
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=10, worker_init_fn=toolbox.worker_init_fn, drop_last=True) for x in ['train', 'val']}

my_metrics = toolbox.Metrics(metric_names=['loss', 'corrects', 'precision', 'recall', 'f1'], num_classes=num_classes)

N = len(dataloaders_dict['val']) + len(dataloaders_dict['train'])

FPS = 0

input_size = torch.Size([3, input_size, input_size])
   
# GFLOPs, n_params = toolbox.count_flops(model=model, device=device, input_size=input_size)
# print("GFLOPS:", GFLOPs)

# print(N)
# exit()
# for i in range(10):
start = time.time()
for phase in ['train', 'val']:
	for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
		inputs = inputs.to(device)
		labels = labels.to(device)
		# _,_,outputs = model(inputs)
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		_, preds = torch.max(outputs, 1)    
	# stats = metrics.classification_report(labels.data.tolist(), preds.tolist(), output_dict = True)
	# stats_out = stats['weighted avg']
				   
	# my_metrics.update(loss=loss, preds=preds, labels=labels, stats_out=stats_out)
	# epoch_metrics = my_metrics.calculate()
	# print()
	# print(phase)
	# print(epoch_metrics)
	# my_metrics.reset()

# FPS calculation
FPS = (N * batch_size) / (time.time() - start)
print("FPS: ", FPS)

##################################


# def get_gpu_utilization():
#     try:
#         nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).decode("utf-8")
#         gpu_utilization = int(re.findall(r'\d+', nvidia_smi_output)[0])
#         return gpu_utilization
#     except Exception as e:
#         print(f"Error fetching GPU utilization: {e}")
#         return None

# # Assuming you have a model, dataloaders_dict, and device defined
# # Start time for FPS calculation
# start = time.time()

# # Initial GPU memory usage
# initial_memory = torch.cuda.memory_allocated(device)
# peak_memory = initial_memory

# # Evaluation loop
# gpu_utilizations = []
# for phase in ['train', 'val']:
#     for idx, (inputs, labels) in enumerate(dataloaders_dict[phase]):
#         inputs = inputs.to(device)

#         with torch.no_grad():  # Ensure no gradients are calculated
#             outputs = model(inputs)

#         # Check and update peak GPU memory usage
#         current_memory = torch.cuda.memory_allocated(device)
#         peak_memory = max(peak_memory, current_memory)

#         # Measure GPU utilization
#         gpu_util = get_gpu_utilization()
#         if gpu_util is not None:
#             gpu_utilizations.append(gpu_util)

# # FPS calculation
# FPS = (N * batch_size) / (time.time() - start)
# print("FPS: ", FPS)

# # GPU Memory Usage
# used_memory = peak_memory - initial_memory
# # print(f"Initial Memory Usage: {initial_memory / (1024**2)} MB")
# print(f"Peak Memory Usage: {peak_memory / (1024**2)} MB")
# # print(f"Memory Used in Evaluation: {used_memory / (1024**2)} MB")

# # Average GPU Utilization
# if gpu_utilizations:
#     average_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations)
#     print(f"Average GPU Utilization: {average_gpu_utilization}%")
# else:
#     print("GPU Utilization data not available.")