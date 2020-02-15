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
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]

        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                if m.max() < 1:
                    continue
                if annotation["iscrowd"]:
                    class_id *= -1
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
            instance_masks.append(m)
            class_ids.append(class_id)

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
    
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    
        


    

if __name__ == "__main__":
    dataset_train = AirbusDataset(DATASET_CONFIG_PATH)
    dataset_train.prepare()
    dataset_train.load_mask(1)
    # model = modellib.MaskRCNN()

    

