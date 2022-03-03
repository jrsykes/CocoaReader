import torch
from detectron2.utils.logger import setup_logger
setup_logger(output='/local/scratch/jrs596/MaskRCNN/models/MaskRCNN_model_ResDesWeights')

import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog#, DatasetCatalog
#from pycocotools.coco import COCO
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
#from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances

#from detectron2.engine import launch
from detectron2.evaluation import COCOEvaluator#, inference_on_dataset
from detectron2.engine import DefaultTrainer
#from detectron2.checkpoint import DetectionCheckpointer


base_path = '/local/scratch/jrs596/MaskRCNN/dat/results'
register_coco_instances("my_dataset_train", {}, base_path + "/annotations/train_combined_instances_default.json", base_path + "/images/train")
register_coco_instances("my_dataset_val", {}, base_path + "/annotations/val_combined_instances_default.json", base_path + "/images/val")
pod_metadata = MetadataCatalog.get("my_dataset_train")

class CheckerHook(HookBase):
  def after_step(self):
    if self.trainer.iter % 10 == 0:
      print(f"Iteration {self.trainer.iter} complete")




itterations = 20#1000
cfg = get_cfg()
cfg.OUTPUT_DIR = '/local/scratch/jrs596/MaskRCNN/models/MaskRCNN_model_ResDesWeights'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

#setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=0, color=True, name="detectron2", abbrev_name=None)
#logger = logging.getLogger(name)
#logger.setLevel(logging.DEBUG)
#logger.propagate = False

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
cfg.MODEL.WEIGHTS = '/local/scratch/jrs596/ResNetFung50_Torch/models/archive/data.pkl'
cfg.SOLVER.IMS_PER_BATCH = 10#20
cfg.SOLVER.MAX_ITER = itterations
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1000 #Default = 512   
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
cfg.INPUT.CROP.ENABLED = True 
input_size = 224 #Dictated by Imagenet weights
cfg.INPUT.MAX_SIZE_TEST = input_size
cfg.INPUT.MAX_SIZE_TRAIN = input_size
cfg.INPUT.MIN_SIZE_TEST = input_size
cfg.INPUT.MIN_SIZE_TEST = input_size
cfg.INPUT.MIN_SIZE_TRAIN = (input_size)
cfg.INPUT['Normalize'] = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
cfg.TEST.EVAL_PERIOD = 1




model = build_model(cfg)
dataloader = build_detection_train_loader(cfg,
mapper=DatasetMapper(cfg, is_train=True), num_workers=4)
optimizer = torch.optim.Adam(get_default_optimizer_params(model),
               lr=0.00025)


def test_and_save_results():
  self._last_eval_results = self.test(self.cfg, self.model)
  return self._last_eval_results

class SimpleTrainer2(SimpleTrainer):
  

trainer = SimpleTrainer(model, data_loader=dataloader, optimizer = optimizer) 
trainer.register_hooks([CheckerHook(), hooks.IterationTimer(), hooks.EvalHook(cfg.TEST.EVAL_PERIOD, trainer.build_hooks().test_and_save_results())])
trainer.train(start_iter=0, max_iter=itterations)


#PATH = '/local/scratch/jrs596/MaskRCNN/models/'
checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
#checkpointer.save("MaskRCNN_model_ResDesWeights")  # save to output/model_999.pth
checkpointer.save("test")  # save to output/model_999.pth


#cfg.MODEL.WEIGHTS = os.path.join(PATH, 'MaskRCNN_model_ResDesWeights.pth')  # path to the model we just trained
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
#predictor = DefaultPredictor(cfg)##

#test_datasets = ["my_dataset_val"]
#evaluator = [COCOEvaluator(test_set, cfg, False) for test_set in test_datasets]
#metrics = DefaultTrainer.test(cfg, model, evaluator)

