import torch
from detectron2.utils.logger import setup_logger
setup_logger()

import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog#, DatasetCatalog
from torchvision import transforms

from detectron2.engine import SimpleTrainer
from detectron2.engine import HookBase
from detectron2.solver import get_default_optimizer_params
from detectron2.engine import hooks
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import build_detection_train_loader
from detectron2.export import Caffe2Model, add_export_config, export_caffe2_model
from detectron2.modeling import build_model
from detectron2.data.datasets import register_coco_instances

from detectron2.evaluation import COCOEvaluator#, inference_on_dataset
from detectron2.engine import DefaultTrainer
import pickle
import copy

base_path = '/local/scratch/jrs596/MaskRCNN/dat/results'
register_coco_instances("my_dataset_train", {}, base_path + "/annotations/train_combined_instances_default.json", base_path + "/images/train")
register_coco_instances("my_dataset_val", {}, base_path + "/annotations/val_combined_instances_default.json", base_path + "/images/val")
pod_metadata = MetadataCatalog.get("my_dataset_train")


    # load pretrained weights from ResDesNet50 
pretrained_model_path = '/local/scratch/jrs596/ResNetFung50_Torch/models/model.pkl'
pretrained_model_wts = pickle.load(open(pretrained_model_path, "rb"))
pretrained_model_wts = copy.deepcopy(pretrained_model_wts['model'])

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = pretrained_model_wts
cfg.MODEL.WEIGHTS = '/local/scratch/jrs596/ResNetFung50_Torch/models/model.pkl'
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 100    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1000   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.INPUT.CROP.ENABLED = True 
input_size = 224 #Dictated by Imagenet weights
cfg.INPUT.MAX_SIZE_TEST = input_size
cfg.INPUT.MAX_SIZE_TRAIN = input_size
cfg.INPUT.MIN_SIZE_TEST = input_size
cfg.INPUT.MIN_SIZE_TEST = input_size
cfg.INPUT.MIN_SIZE_TRAIN = (input_size)
cfg.INPUT['Normalize'] = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

print(cfg.MODEL.WEIGHTS)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


#PATH = '/local/scratch/jrs596/MaskRCNN/models/'
checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
#checkpointer.save("MaskRCNN_model_ResDesWeights")  # save to output/model_999.pth
checkpointer.save("MRCNN_ResDesNetWeights")  # save to output/model_999.pth

## Model evaluation

#cfg.MODEL.WEIGHTS = os.path.join(PATH, 'MaskRCNN_model_ResDesWeights.pth')  # path to the model we just trained
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
#predictor = DefaultPredictor(cfg)##

#test_datasets = ["my_dataset_val"]
#evaluator = [COCOEvaluator(test_set, cfg, False) for test_set in test_datasets]
#metrics = DefaultTrainer.test(cfg, model, evaluator)

