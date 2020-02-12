import os
import sys
import random
import math
import json
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# from mrcnn.config import Config
from mrcnn import utils
# import mrcnn.model as modellib
# from mrcnn import visualize
# from mrcnn.model import log

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils


# Directory to save logs and trained model
MODEL_DIR = "./logs"

# Local path to trained weights file
COCO_MODEL_PATH = "mask_rcnn_coco.h5"

# Airbus dataset config file
DATASET_CONFIG_PATH = "./data/annotations/instances_ships_train2018.json"


class AirbusDataset(utils.Dataset):
    def __init__(self, data_config):
        super(AirbusDataset, self).__init__()

        self.data_config = data_config
        coco = COCO(self.data_config)
        image_dir = "./data/train_v2"
        class_ids = sorted(coco.getCatIds())
        
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            image_ids = list(set(image_ids))
        else:
            image_ids = list(coco.imgs.keys())

        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])
        
        for i in image_ids:
            self.add_image(source = "airbus",
                            image_id = i,
                            path = os.path.join(image_dir, coco.imgs[i]["file_name"]),
                            width = coco.imgs[i]["width"],
                            height = coco.imgs[i]["height"],
                            annotations = coco.loadAnns(coco.getAnnIds(
                                                            imgIds = [i],
                                                            catIds = class_ids,
                                                            iscrowd = None)))

    def load_mask(self, image_id):
        ''' Load instance masks for the given image
            Returns:
                mask: A bool array of shape [height, width, instance count]
                    with one mask per instance.
                class_ids: a 1D array of class IDs of the instance masks
        '''

        image_info = self.image_info[image_id]
        isinstance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask

        # for annotation in annotations:
        #     class_id = self.ma



    def parse_config(self):
        with open(self.data_config) as json_file:
            configs = json.load(json_file)
        print(configs.keys())
        self.num_of_images = len(configs["images"])
        print(configs["images"][0])
        


    

if __name__ == "__main__":
    dataset_train = AirbusDataset(DATASET_CONFIG_PATH)
    dataset_train.load_mask(1)
    # model = modellib.MaskRCNN()

    

