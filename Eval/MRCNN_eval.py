
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from torchvision import datasets, models, transforms


base_path = '/local/scratch/jrs596/MaskRCNN/dat/results'
register_coco_instances("my_dataset_val", {}, base_path + "/annotations/val_combined_instances_default.json", base_path + "/images/val")
pod_metadata = MetadataCatalog.get("my_dataset_train")

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg = get_cfg()
cfg.MODEL.WEIGHTS = '/local/scratch/jrs596/MaskRCNN/models/MaskRCNN_model_ResDesWeights.pth'  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)#
"""Then, we randomly select several samples to visualize the prediction results."""#
from detectron2.utils.visualizer import ColorMode
#dataset_dicts = get_balloon_dicts("balloon/val")
#for d in random.sample(dataset_dicts, 3):    
im = cv2.imread("/local/scratch/jrs596/MaskRCNN/dat/results/images/val/BPR1645087278.11426674.jpeg")
outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
v = Visualizer(im[:, :, ::-1],
                   metadata=pod_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

print(out)
#cv2.imshow(out.get_image()[:, :, ::-1])#
#"""We can also evaluate its performance using AP metric implemented in COCO API.
#This gives an AP of ~70. Not bad!
#"""#
#from detectron2.evaluation import COCOEvaluator, inference_on_dataset
#from detectron2.data import build_detection_test_loader
#evaluator = COCOEvaluator("my_dataset_val", output_dir="./output")
#val_loader = build_detection_test_loader(cfg, "my_dataset_val")
#print(inference_on_dataset(predictor.model, val_loader, evaluator))
## another equivalent way to evaluate the model is to use `trainer.test`