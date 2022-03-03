# check pytorch installation: 
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
#assert torch.__version__.startswith("1.9")   # please manually install torch 1.9 if Colab changes its default version

#!nvidia-smi

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#from google.colab import drive
#drive.mount('/content/drive')

from pycocotools.coco import COCO
#import numpy as np
#import skimage.io as io
#import matplotlib.pyplot as plt

base_path = '/local/scratch/jrs596/MaskRCNN/dat/results'
dataDir= base_path + '/images/'
dataType='val2017'
annFile= base_path + '/annotations/train_combined_instances_default.json'.format(dataDir,dataType)
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['Black pod rot', 'Witches broom disease', 'Frosty pod rot']);
#imgIds = coco.getImgIds(catIds=catIds );
#imgIds = coco.getImgIds(imgIds = [21])
#img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

#I = io.imread('%s/%s'%(dataDir,img['file_name']))
#plt.axis('off')
#plt.imshow(I)
#plt.show()

#plt.imshow(I); plt.axis('off')
#annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
#anns = coco.loadAnns(annIds)
#coco.showAnns(anns)

#Augmentations

#from detectron2.data import transforms as T
## Define a sequence of augmentations:
#augs = T.AugmentationList([
#    T.RandomBrightness(0.7, 1.3),
#    #T.RandomFlip(prob=1, vertical=False),
#    T.RandomContrast(0.8,1.2),
#    T.RandomSaturation(0.5,1.5),
#    T.RandomLighting(100)
#])  # type: T.Augmentation


#input = T.AugInput(I, sem_seg=anns)
#transform = augs(input)  # type: T.Transform
#image_transformed = input.image  # new image
#sem_seg_transformed = input.sem_seg  # new semantic segmentation#
#

#plt.imshow(image_transformed); plt.axis('off')
#coco.showAnns(sem_seg_transformed)

from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, base_path + "/annotations/train_combined_instances_default.json", base_path + "/images/train")
register_coco_instances("my_dataset_val", {}, base_path + "annotations/val_combined_instances_default.json", base_path + "/images/val")
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
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 42   
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 3
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

model = build_model(cfg)




#import albumentations as A
#transform = A.Compose([
#    #A.HorizontalFlip(p=0.5),
#    A.RandomBrightnessContrast(p=0.5),
#    A.HueSaturationValue(-10,10),
#    A.RandomShadow(),
#    A.Sharpen(),
#    A.CLAHE(),
#    A.Rotate(20),
#    #A.GridDropout(ratio=0.4, p=0.2)   
#])

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

#dataloader = build_detection_train_loader(cfg,
  # mapper=DatasetMapper(cfg, is_train=True, augmentations=[
 #     T.Resize((800, 800))
#   ]), num_workers=2)

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
    

#class ProfilerHook(HookBase):
 # def after_step(self):
  #  hooks.TorchProfiler(self, cfg.OUTPUT_DIR)


trainer = SimpleTrainer(model, data_loader=dataloader, optimizer = optimizer) 
trainer.register_hooks([CheckerHook()])


trainer.train(start_iter=0, max_iter=100)


checkpointer = DetectionCheckpointer(model, save_dir="output")
checkpointer.save("/local/scratch/jrs596/MaskRCNN/models/MRCNN_subset_model")  # save model

from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "/local/scratch/jrs596/MaskRCNN/models/MRCNN_subset_model.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

test_datasets = ["my_dataset_val"]
evaluator = [COCOEvaluator(test_set, cfg, False) for test_set in test_datasets]
metrics = DefaultTrainer.test(cfg, model, evaluator)

# Look at training curves in tensorboard:
#%load_ext tensorboard
#%tensorboard --logdir output
#%tensorboard --logdir /content/output/log


#from detectron2.utils.visualizer import ColorMode
#dataset_dicts = get_balloon_dicts("balloon/val")
#for d in random.sample(dataset_dicts, 3):    
#im = cv2.imread("/content/drive/MyDrive/MaskR-CNN_annotated_subset/annotated_data_wholtree_subset/test/images/0415-diseased-cacao-812x1200.jpg")#

#outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#v = Visualizer(im[:, :, ::-1],
#                   metadata=pod_metadata, 
#                   scale=0.5, 
#                   instance_mode=ColorMode.SEGMENTATION   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#    )
#out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#cv2_imshow(out.get_image()[:, :, ::-1])
