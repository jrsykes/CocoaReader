import sys
import time
import warnings  # To disable warnings on export to ONNX.
import zipfile
from pathlib import Path
import logging

import torch
import nncf  # Important - should be imported directly after torch.

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

from nncf.common.utils.logger import set_log_level
set_log_level(logging.ERROR)  # Disables all NNCF info and warning messages.
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from openvino.runtime import Core
from torch.jit import TracerWarning

sys.path.append("../utils")
from notebook_utils import download_file

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

MODEL_DIR = Path("/local/scratch/jrs596/dat/models")
OUTPUT_DIR = Path("/local/scratch/jrs596/dat")
DATA_DIR = Path("/local/scratch/jrs596/dat/split_cocoa_images/")
BASE_MODEL_NAME = "CocoaNet18_DN"
image_size = 224

#OUTPUT_DIR.mkdir(exist_ok=True)
#MODEL_DIR.mkdir(exist_ok=True)
#DATA_DIR.mkdir(exist_ok=True)

# Paths where PyTorch, ONNX and OpenVINO IR models will be stored.
fp32_pth_path = Path(MODEL_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".pth")
fp32_onnx_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_fp32")).with_suffix(".onnx")
fp32_ir_path = fp32_onnx_path.with_suffix(".xml")
int8_onnx_path = Path(OUTPUT_DIR / (BASE_MODEL_NAME + "_int8")).with_suffix(".onnx")
int8_ir_path = int8_onnx_path.with_suffix(".xml")

