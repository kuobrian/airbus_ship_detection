import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from model_UResnet34 import Resnet34Unet
from imgaug import augmenters as iaa
from train import AirbusDataset
from skimage.morphology import binary_opening, disk, label
from utils import *



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
    if os.path.exists("./uresnet34_best.pt"):
        model.load_state_dict(torch.load("./uresnet34_best.pt"))
    model.eval()

    ship_dir = "./data"
    test_df = pd.read_csv(os.path.join(ship_dir, "sample_submission_v2.csv"))

    test_tfs = transforms.Compose([ iaa.Sequential([
                                        iaa.size.Resize({"height": 512, "width": 512})
                                        ]).augment_image
                            ])


    test_dataset = AirbusDataset(ship_dir, test_df, transform = test_tfs, mode="test")

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=1)

    submission = {'ImageId': [], 'EncodedPixels': []}


    for i, (imgO, image_name) in enumerate(test_loader, 0):
        imgO = imgO.to(device)
        output = model(imgO)

        for i, fname in enumerate(image_name):
            mask = F.sigmoid(output[i, 0]).data.cpu().numpy()
            mask = binary_opening(mask > 0.5, disk(2))

            mask = np.array(mask).astype(np.bool)
            labels = label(mask)
            encodings = [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]
            if len(encodings) > 0:
                for encoding in encodings:
                    submission['ImageId'].append(fname)
                    submission['EncodedPixels'].append(encoding)
            else:
                submission['ImageId'].append(fname)
                submission['EncodedPixels'].append(None)

        assert(0)
    
    submission_df = pd.DataFrame(submission, columns=['ImageId', 'EncodedPixels'])
    submission_df.to_csv('submission.csv', index=False)
    submission_df.sample(10)

    # output = ((output > 0).float()) * 255
    # imgO = imgO.data.cpu()
    # output = output.data.cpu()

    # grid_imgs = torchvision.utils.make_grid(imgO, nrow=1)
    # grid_labels = torchvision.utils.make_grid(labels, nrow=1)
    # grid_output = torchvision.utils.make_grid(output, nrow=1)

    # imshow_gt_out(grid_imgs, grid_labels, grid_output)
    # plt.show()

