import os
import sys
import json
import datetime
import numpy as np
import random
import skimage.draw
import matplotlib.pyplot as plt
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import utils
from mrcnn.config import Config
from mrcnn.visualize import display_images
from matplotlib import patches,  lines

class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "balloon"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

class BalloonDataset():
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                return
        self.class_info.append({"source": source, "id": class_id, "name": class_name})

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

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_balloon(self, dataset_dir, subset):
        self.add_class("balloon", 1, "balloon")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "balloon",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)



if __name__ == "__main__":
    config = BalloonConfig()
    BALLOON_DIR = os.path.join("./data", "balloon")
    
    dataset = BalloonDataset()
    dataset.load_balloon(BALLOON_DIR, "train")

    dataset.prepare()
    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # Load and display random samples
    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        # visualize.display_top_masks(image, mask, class_ids, dataset.class_names)
    
    # Load random image and mask.
    image_id = random.choice(dataset.image_ids)
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    # Compute Bounding box
    bbox = utils.extract_bboxes(mask)

    # Display image and additional stats
    print("image_id ", image_id, dataset.image_reference(image_id))
    # visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


    image_id = np.random.choice(dataset.image_ids, 1)[0]
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape

    image, window, scale, padding, _ = utils.resize_image(image,
                                                        min_dim=config.IMAGE_MIN_DIM, 
                                                        max_dim=config.IMAGE_MAX_DIM,
                                                        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding)
    bbox = utils.extract_bboxes(mask)
    # visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
    
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
                                dataset, config, image_id, use_mini_mask=False)
    
    # visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)

    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, 
                                            config.RPN_ANCHOR_RATIOS,
                                            backbone_shapes,
                                            config.BACKBONE_STRIDES, 
                                            config.RPN_ANCHOR_STRIDE)
    # Print summary of anchors
    num_levels = len(backbone_shapes)
    anchors_per_cell = len(config.RPN_ANCHOR_RATIOS)
    print("Count: ", anchors.shape[0])
    print("Scales: ", config.RPN_ANCHOR_SCALES)
    print("ratios: ", config.RPN_ANCHOR_RATIOS)
    print("Anchors per Cell: ", anchors_per_cell)
    print("Levels: ", num_levels)
    anchors_per_level = []
    for l in range(num_levels):
        num_cells = backbone_shapes[l][0] * backbone_shapes[l][1]
        anchors_per_level.append(anchors_per_cell * num_cells // config.RPN_ANCHOR_STRIDE**2)
        print("Anchors in Level {}: {}".format(l, anchors_per_level[l]))
    
    ## Visualize anchors of one cell at the center of the feature map of a specific level

    # Load and draw random image
    image_id = np.random.choice(dataset.image_ids, 1)[0]
    image, image_meta, _, _, _ = modellib.load_image_gt(dataset, config, image_id)
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    levels = len(backbone_shapes)

    for level in range(levels):
        colors = visualize.random_colors(levels)
        # Compute the index of the anchors at the center of the image
        level_start = sum(anchors_per_level[:level]) # sum of anchors of previous levels
        level_anchors = anchors[level_start:level_start+anchors_per_level[level]]
        print("Level {}. Anchors: {:6}  Feature map Shape: {}".format(level, level_anchors.shape[0], 
                                                                    backbone_shapes[level]))
        center_cell = backbone_shapes[level] // 2
        center_cell_index = (center_cell[0] * backbone_shapes[level][1] + center_cell[1])
        level_center = center_cell_index * anchors_per_cell 
        center_anchor = anchors_per_cell * (
            (center_cell[0] * backbone_shapes[level][1] / config.RPN_ANCHOR_STRIDE**2) \
            + center_cell[1] / config.RPN_ANCHOR_STRIDE)
        level_center = int(center_anchor)

        # Draw anchors. Brightness show the order in the array, dark to bright.
        for i, rect in enumerate(level_anchors[level_center:level_center+anchors_per_cell]):
            y1, x1, y2, x2 = rect
            p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, facecolor='none',
                                edgecolor=(i+1)*np.array(colors[level]) / anchors_per_cell)
            ax.add_patch(p)
    # plt.show()
    random_rois = 2000
    g = modellib.data_generator(dataset, config, shuffle=True, random_rois=random_rois, 
                                batch_size=4, detection_targets=True)

