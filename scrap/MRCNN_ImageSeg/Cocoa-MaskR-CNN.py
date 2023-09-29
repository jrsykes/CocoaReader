# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from pycocotools.coco import COCO
#import tensorflow


base_path = '/local/scratch/jrs596/MaskRCNN/dat/results'
dataDir= base_path + '/images/'
dataType='val2017'
annFile= base_path + '/annotations/train_combined_instances_default.json'.format(dataDir,dataType)
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['Black pod rot', 'Witches broom disease', 'Frosty pod rot']);


from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, base_path + "/annotations/train_combined_instances_default.json", base_path + "/images/train")
register_coco_instances("my_dataset_val", {}, base_path + "/annotations/val_combined_instances_default.json", base_path + "/images/val")
#register_coco_instances("my_dataset_test", {}, base_path + "/test/annotations/instances_default.json", base_path +  "/test/images")

pod_metadata = MetadataCatalog.get("my_dataset_train")

from detectron2.export import Caffe2Model, add_export_config, export_caffe2_model
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger

#setup_logger()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #Default = 512   
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


model = build_model(cfg)


from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate, RandomCrop
)

transform = Compose([
    Rotate(limit=40),
    RandomBrightness(limit=0.1),
    JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
    HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    RandomContrast(limit=0.2, p=0.5),
    HorizontalFlip(),
    RandomCrop(height=50, width=50, always_apply=False, p=0.2),
])

from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader


dataloader = build_detection_train_loader(cfg,
   mapper=DatasetMapper(cfg, is_train=True), num_workers=2)


from detectron2.engine import SimpleTrainer
from detectron2.engine import HookBase
from detectron2.solver import get_default_optimizer_params
from detectron2.engine import hooks
from detectron2.checkpoint import DetectionCheckpointer

optimizer = torch.optim.Adam(get_default_optimizer_params(model),
               lr=0.00025)



class CheckerHook(HookBase):
  def after_step(self):
    if self.trainer.iter % 10 == 0:
      print(f"Iteration {self.trainer.iter} complete")
    

trainer = SimpleTrainer(model, data_loader=dataloader, optimizer = optimizer) 
trainer.register_hooks([CheckerHook()])


trainer.train(start_iter=0, max_iter=1000)



#torch.save(trainer.model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'MRCNN_cocoa_model.pth'))

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer
from detectron2.checkpoint import DetectionCheckpointer
#DetectionCheckpointer(model).load(file_path_or_url)  # load a file, usually from cfg.MODEL.WEIGHTS

PATH = '/local/scratch/jrs596/MaskRCNN/models/'
checkpointer = DetectionCheckpointer(trainer.model, save_dir=PATH)
checkpointer.save("MaskRCNN_model")  # save to output/model_999.pth


cfg.MODEL.WEIGHTS = os.path.join(PATH, 'MaskRCNN_model.pth')  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)##

test_datasets = ["my_dataset_val"]
evaluator = [COCOEvaluator(test_set, cfg, False) for test_set in test_datasets]
metrics = DefaultTrainer.test(cfg, model, evaluator)

