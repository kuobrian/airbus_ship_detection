import os
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


    
    def build(self, mode, config):
        pass
