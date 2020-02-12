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

        # self.parse_config()

    def load_airbus(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        
        ''' Load a subset of the COCO dataset.
            dataset_dir: The root directory of the COCO dataset.
            subset: What to load (train, val, minival, valminusminival)
            year: What dataset year to load (2014, 2017) as a string, not an integer
            class_ids: If provided, only loads images that have the given classes.
            class_map: TODO: Not implemented yet. Supports maping classes from
                different datasets to the same class ID.
            return_coco: If True, returns the COCO object.
            auto_download: Automatically download and unzip MS-COCO images and annotations
        '''
        

    def parse_config(self):
        with open(self.data_config) as json_file:
            configs = json.load(json_file)
        print(configs.keys())
        self.num_of_images = len(configs["images"])
        print(configs["images"][0])
        


    

if __name__ == "__main__":
    dataset_train = AirbusDataset(DATASET_CONFIG_PATH)
    # model = modellib.MaskRCNN()

    

