import torch
from torch.utils.mobile_optimizer import optimize_for_mobile


traced_script_module_optimized = torch.jit.load("/home/jamiesykes/Documents/ModelZoo/CocoaNet18_quantised_mobile.pth")

traced_script_module_optimized._save_for_lite_interpreter("~/AndroidStudioProjects/CocoaReader/app/src/main/assets/CocoaNet18_quantised_mobile.ptl")

