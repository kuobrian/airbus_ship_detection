import os
from mrcnn import visualize
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import sys

from mrcnn import utils



# Import airbus_dataset
sys.path.append(os.path.abspath(os.getcwd()))  # To find local version of the library
# import airbus_dataset

############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):

    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(image,
                                                            min_dim=config.IMAGE_MIN_DIM,
                                                            min_scale=config.IMAGE_MIN_SCALE,
                                                            max_dim=config.IMAGE_MAX_DIM,
                                                            mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS
        
        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)
    
    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)
    
    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1


    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask
    
def compute_backbone_shapes(config, image_shape):
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)
    assert config.BACKBONE in ["resnet50", "resnet101"]
    
    return np.array([[int(math.ceil(image_shape[0] / stride)),
                        int(math.ceil(image_shape[1] / stride))] 
                            for stride in config.BACKBONE_STRIDES])


############################################################
#  Data Generator
############################################################
def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
 
    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    print(image_shape)
    print(anchors.shape)
    print(gt_class_ids)
    print(gt_boxes)
    
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    print(rpn_bbox.shape)

def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   random_rois=0, batch_size=1, detection_targets=False,
                   no_augmentation_sources=None):
    batch_size = 0
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []
    
    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)
    # while True:
    #     try:
    image_index = (image_index + 1) % len(image_ids)
    if shuffle and image_index == 0:
        np.random.shuffle(image_ids)

    # Get GT bounding boxes and masks for image.
    image_id = image_ids[image_index]
    if dataset.image_info[image_id]['source'] in no_augmentation_sources:
        image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
        load_image_gt(dataset, config, image_id, augment=augment,
                    augmentation=None,
                    use_mini_mask=config.USE_MINI_MASK)
    else:
        image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
        load_image_gt(dataset, config, image_id, augment=augment,
                    augmentation=augmentation,
                    use_mini_mask=config.USE_MINI_MASK)
    
    # if not np.any(gt_class_ids > 0):
    #     continue
    
    # RPN Targets
    rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                            gt_class_ids, gt_boxes, config)

            

        # except (GeneratorExit, KeyboardInterrupt):
        #     raise
        # except:
        #     # Log it and skip the image
        #     logging.exception("Error processing image {}".format(
        #         dataset.image_info[image_id]))
        #     error_count += 1
        #     if error_count > 5:
        #         raise






class MaskRCNN():
    def __init__(self, mode, config, model_dir):
        '''
        mode: "train" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        '''
        assert mode in ["train" , "inference"]
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """
            Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ["training", "inference"]
        
        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")


        # Inputs
        input_image = KL.Input(
            shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
        input_image_meta = KL.Input(shape=conofig.IMAGE_META_SIZE,
                                    name="input_image_meta")
        
        if mode == "train":
            # RPN GT
            pass
        elif mode == "inference":
            pass

    
    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        self.epoch = 0
        now = datetime.datetime.now()
        
        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")


# if __name__ == "__main__":
    # ROOT_DIR = os.path.abspath("../")
    # COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    # MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # DATASET_CONFIG_PATH = "./data/annotations/instances_ships_train2018.json"
    # AIRBUS_DIR = "./data/train_v2"

    # # Dataset
    # dataset = airbus_dataset.AirbusDataset(DATASET_CONFIG_PATH)
    # dataset.load_data(DATASET_CONFIG_PATH)
    # dataset.prepare()
    # dataset.load_mask(1)

    # config = airbus_dataset.AirbusConfig()
    # print("Image Count: {}".format(len(dataset.image_ids)))
    # print("Class Count: {}".format(dataset.num_class))
    # for i, info in enumerate(dataset.class_info):
    #     print("{:3}. {:50}".format(i, info['name']))


    # image_ids = np.random.choice(dataset.image_ids, 4)
    # for image_id in image_ids:
    #     image = dataset.load_image(image_id)
    #     mask, class_ids = dataset.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


    # # Create model object in inference mode.
    # model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # # # Load weights trained on MS-COCO
    # # model.load_weights(COCO_MODEL_PATH, by_name=True)