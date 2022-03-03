import pickle
import copy


    # load pretrained weights from ResDesNet50 

pretrained_model_path = '/local/scratch/jrs596/ResNetFung50_Torch/models/model.pkl'
pretrained_model_wts = pickle.load(open(pretrained_model_path, "rb"))
pretrained_model_wts = copy.deepcopy(pretrained_model_wts['model'])

imagenet_model_path = '/local/scratch/jrs596/ResNetFung50_Torch/models/imagenet_model_final_f10217.pkl'    
imagenet_model_wts = pickle.load(open(imagenet_model_path, "rb"))
imagenet_model_wts = copy.deepcopy(imagenet_model_wts['model'])




#for key, value in pretrained_model_wts.items():
#	if 'module.layer1.0.conv1.weight' in key:
#		print(value.shape)

for key, value in imagenet_model_wts.items():
	if 'backbone.bottom_up.res2.0.shortcut.weight' in key:
		print(value.shape)
