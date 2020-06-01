import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from resnet34Unet import Resnet34Unet
from utils import *
from train import *

def imshow_gt_out(img, mask_gt, mask_out):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    mask_gt = mask_gt.numpy().transpose((1, 2, 0))
    mask_gt = np.clip(mask_gt, 0, 1)

    mask_out = mask_out.numpy().transpose((1, 2, 0))
    mask_out = np.clip(mask_out, 0, 1)

    fig, axs = plt.subplots(1,3, figsize=(10,30))
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title("Input image")
    axs[1].imshow(mask_gt)
    axs[1].axis('off')
    axs[1].set_title("Ground truth")
    axs[2].imshow(mask_out)
    axs[2].axis('off')
    axs[2].set_title("Model output")
    plt.subplots_adjust(wspace=0, hspace=0)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Resnet34Unet(3, 1).to(device)
    model.load_state_dict(torch.load("./uresnet34_best.pt"))

    valid__path = "./valid_set.csv"
    ship_dir = "../data"


    valid_df = pd.read_csv(valid__path)



    val_transform = DualCompose([CenterCrop((512,512,3))])

    valid_dataset = AirbusDataset(ship_dir, valid_df, transform = val_transform)

    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=1)

                            
    for i, (imgO, labels) in enumerate(valid_loader, 0):
        imgO = imgO.to(device)       # [B, 3, H, W]
        labels = labels  # [B, H, W] with class indices (0, 1)
        output = model(imgO)
        # print(imgO.shape, labels.shape, output.shape)
        output = ((output > 0).float()) * 255
        print(output.shape)

        imgO = imgO.data.cpu()
        output = output.data.cpu()
        
        print(labels.shape)
        print(torch.max(labels))
        print(torch.min(labels))


        grid_imgs = torchvision.utils.make_grid(imgO, nrow=1)
        grid_labels = torchvision.utils.make_grid(labels, nrow=1)
        grid_output = torchvision.utils.make_grid(output, nrow=1)
        
        print(grid_labels.shape)
        print(torch.max(grid_labels))
        print(torch.min(grid_labels))
        



        # imshow_gt_out(grid_imgs, grid_labels, grid_output)
        plt.show()

        assert(0)
