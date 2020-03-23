import os
import sys
import random
import math
import json
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import visualize
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

class AirbusConfig(Config):
    NAME = "airbus"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class AirbusDataset():
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
               
    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                return
        self.class_info.append({"source": source,
                                "id": class_id,
                                "name": class_name})

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {"id": image_id, "source": source, "path": path}
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info["path"]

    def prepare(self, class_map=None):
        def clean_name(name):
            return ",".join(name.split(",")[:1])

        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        for source in self.sources:
            self.source_class_ids[source] = []
            for i, info in enumerate(self.class_info):
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']
        
    def load_airbus(self, dataset_dir, subset="train"):      
        airbus_set = COCO("{}/annotations/instances_ships_{}2018.json".format(dataset_dir, subset))
        image_dir = "{}/{}_v2".format(dataset_dir, subset)
        class_ids = sorted(airbus_set.getCatIds())
        
        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(airbus_set.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(airbus_set.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("airbus", i, airbus_set.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "airbus", image_id=i,
                path=os.path.join(image_dir, airbus_set.imgs[i]['file_name']),
                width=airbus_set.imgs[i]["width"],
                height=airbus_set.imgs[i]["height"],
                annotations=airbus_set.loadAnns(airbus_set.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
                


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

    def annToMask(self, ann, h, w):
        rle = self.annToRLE(ann, h, w)
        
        m = maskUtils.decode(rle)
        return m 
    
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

        # mask = np.zeros((image_info["height"], image_info["width"]))
        for ann in annotations:
            class_id =self.map_source_class_id( "airbus.{}".format(ann['category_id']))
            if class_id:
                m = self.annToMask(ann, image_info["height"], image_info["width"])
                # mask += m
                if m.max() < 1:
                    continue
                isinstance_masks.append(m)
                class_ids.append(class_id)
        if class_ids:
            mask = np.stack(isinstance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        # matplotlib.use('TkAgg')
        # plt.imshow(mask*255)
        # plt.show()
        
    def load_image(self, image_id):
        """ Load the specified image and return a [H,W,3] Numpy array. """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        return self.image_info[image_id]["path"]

    
if __name__ == "__main__":
    # Dataset
    airbus_dataset = AirbusDataset()
    airbus_dataset.load_airbus("./data")
    # airbus_dataset.prepare()
    # airbus_dataset.load_mask(1)

    # Model
    # model = modellib.MaskRCNN()

    

