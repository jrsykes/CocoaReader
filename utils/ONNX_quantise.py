from PIL import Image
import numpy as np
import onnxruntime
import torch
#from torchvision import models, datasets, transforms

from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import os
import onnx
import onnxoptimizer
from onnxsim import simplify

image_height = 277
image_width = 277
model_name = 'CocoaNet18_V1.3'
root = '/local/scratch/jrs596/dat/models/'
img_path = '/local/scratch/jrs596/dat/split_cocoa_images/val/FPR/caca0-3JPG.jpg'


# Convert model to ONNX
model = torch.load(os.path.join(root, model_name + '.pth'), map_location=torch.device('cpu'))

def preprocess_image(image_path, height, width, channels=3):
    image = Image.open(image_path)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    image_data = np.asarray(image).astype(np.float32)  
    image_data = image_data.transpose([2, 0, 1]) # transpose to CHWll -h 
    for channel in range(image_data.shape[0]):
        image_data[channel, :, :] = (image_data[channel, :, :] / 255)
    image_data = np.expand_dims(image_data, 0)
    return image_data

#image_file = "/local/scratch/jrs596/dat/FPR.jpg"

x = preprocess_image(img_path, image_height, image_width)
x = torch.tensor(x)
torch_out = model(x)

# Export the model
torch.onnx.export(model,                 # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  os.path.join(root, model_name + '.onnx'), # model input (or a tuple for multiple inputs)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names

##############################################

# Read the categories
with open("/local/scratch/jrs596/dat/classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

session_fp32 = onnxruntime.InferenceSession(os.path.join(root, model_name + '.onnx'))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_sample(session, image_file, categories):
    input_ = {'input':preprocess_image(image_file, image_height, image_width)}
    output = session.run([], {'input':preprocess_image(image_file, image_height, image_width)})[0]
    output = output.flatten()
    #output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:6]
    for catid in top5_catid:
        print(categories[catid], output[catid])

print('Output from float32 onnx converted model:\n')
run_sample(session_fp32, img_path, categories)
print()

###############################
# Compare with original .pth model
print('Output from original float32 pytorch model:\n')

#model = torch.load(os.path.join(root, model_name + '.pth'), map_location=torch.device('cpu'))#

x = preprocess_image(img_path, image_height, image_width)
x = torch.tensor(x)
out = model(x)#
for index, i in enumerate(out[0]):
    print(categories[index], i.item())

######################


def preprocess_func(images_folder, height, width, size_limit=0):
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + '/' + image_name
        image_data = preprocess_image(image_filepath, height, width)
        unconcatenated_batch_data.append(image_data)
    print('Aqui')
    ex_batch = np.expand_dims(unconcatenated_batch_data, axis=0)
    batch_data = np.concatenate(ex_batch, axis=0)
    
    return batch_data

#image_folder = '/local/scratch/jrs596/dat/subset_cocoa_images'
#preprocess_func(image_folder, image_height, image_width, size_limit=0)


class MobilenetDataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder):
        self.image_folder = calibration_image_folder
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            nhwc_data_list = preprocess_func(self.image_folder, image_height, image_width, size_limit=0)
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{'input': nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)


## change it to your real calibration data set
calibration_data_folder = "/local/scratch/jrs596/dat/subset_cocoa_images"
dr = MobilenetDataReader(calibration_data_folder)
quantize_static(os.path.join(root, model_name + '.onnx'),
                os.path.join(root, model_name + '_uint8.onnx'),
                dr)#
#######################################
##Optimize the model
onnx_model = onnx.load(os.path.join(root, model_name + '_uint8.onnx'))
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
optimized_model = onnxoptimizer.optimize(onnx_model, passes)#
onnx.save(optimized_model, os.path.join(root, model_name + '_uint8-opt.onnx'))#


# load your predefined ONNX model
model = onnx.load(os.path.join(root, model_name + '_uint8-opt.onnx'))

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, os.path.join(root, model_name + '_uint8-opt_simp.onnx'))
###########################################

print('ONNX full precision model size (MB):', os.path.getsize(os.path.join(root, model_name + '.onnx'))/(1024*1024))
print('ONNX quantized model size (MB):', os.path.getsize(os.path.join(root, model_name + '_uint8-opt.onnx'))/(1024*1024))#
print('ONNX simplified model size (MB):', os.path.getsize(os.path.join(root, model_name + '_uint8-opt_simp.onnx'))/(1024*1024))#

print()
print('Output from Int8 converted ONNX model:\n')
session_quant = onnxruntime.InferenceSession(os.path.join(root, model_name + '_uint8-opt.onnx'))
run_sample(session_fp32, img_path, categories)


print()
print('Output from Int8 converted, quantised and simplifed ONNX model:\n')
session_quant = onnxruntime.InferenceSession(os.path.join(root, model_name + '_uint8-opt_simp.onnx'))
run_sample(session_fp32, img_path, categories)


# use model_simp as a standard ONNX model object

#Convert onnx model to ort format.
#Run in BASH
#python -m onnxruntime.tools.convert_onnx_models_to_ort --optimization_level basic ./