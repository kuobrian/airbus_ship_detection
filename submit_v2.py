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
from skimage.morphology import binary_opening, disk, label
from imgaug import augmenters as iaa
from utils import *
from train import AirbusDataset


def vis_validate_data(device, test_img_size):
    ship_dir = "./data"
    train_path = os.path.join(ship_dir, "train_v2")
    test_path = os.path.join(ship_dir, "test_v2")

    if not os.path.exists("./processing_data/unique.csv"):
        unique_img_ids, masks = unique_images(ship_dir, train_path, test_path)
        unique_img_ids.to_csv("./processing_data/unique.csv")
    else:
        unique_img_ids = pd.read_csv("./processing_data/unique.csv")
        masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))

    # balanced training data
    # train_df, valid_df = balanced_images(masks, unique_img_ids)

    # drop images without ships
    train_df, valid_df = drop_img_without_ships(masks, unique_img_ids)

    tfs = transforms.Compose([ iaa.Sequential([
                                        iaa.size.Resize({"height": test_img_size, "width": test_img_size})
                                        ]).augment_image
                            ])

    valid_dataset = AirbusDataset(ship_dir, valid_df, transform = tfs)
    # assert(0)
    val_loader = DataLoader(dataset=valid_dataset,
                            batch_size=4,
                            shuffle=False,
                            num_workers=4)
    
    model = Resnet34Unet(3, 1).to(device)
    if os.path.exists("./weights/uresnet34_best.pt"):
        print('Load pre-trained weights !! ')
        checkpoint = torch.load("./weights/uresnet34_best.pt")
        model.load_state_dict(checkpoint['weights'])
        Iou = checkpoint['Iou']
        val_loss = checkpoint['dev_loss']
    print(val_loss, Iou)
    model.eval()

    for batch_idx, (imgO, labels) in enumerate(val_loader):
        imgO = imgO.to(device)
        output = model(imgO)
        output = ((output > 0).float()) * 255
        imgO = imgO.data.cpu()
        output = output.data.cpu()

        grid_imgs = torchvision.utils.make_grid(imgO, nrow=1)
        grid_labels = torchvision.utils.make_grid(labels, nrow=1)
        grid_output = torchvision.utils.make_grid(output, nrow=1)

        imshow_gt_out(grid_imgs, grid_labels, grid_output)
        plt.show()


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
    test_img_size = 384 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # vis_validate_data(device, test_img_size)
    # assert(0)

    model = Resnet34Unet(3, 1).to(device)


    # PATH = "./weights/uresnet34_best.pt"
    PATH = "./weights/CUResnet34_best.pt"


    if os.path.exists(PATH):
        print('Load pre-trained weights !! ')
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['weights'])
        Iou = checkpoint['Iou']
        val_loss = checkpoint['dev_loss']
    print(val_loss, Iou)
   
    model.eval()

    ship_dir = "./data"
    test_df = pd.read_csv(os.path.join(ship_dir, "sample_submission_v2.csv"))
    
    test_ships_df = pd.read_csv("./exits_ships_940_256.csv")
    ships_group = test_ships_df.groupby("ships")

    non_ships_df = ships_group.get_group(0)
    ships_df = ships_group.get_group(1)

    non_ships_df["EncodedPixels"] = None
    non_ships_df = non_ships_df.drop(["ships"], axis=1)


    test_tfs = transforms.Compose([ iaa.Sequential([
                                        iaa.size.Resize({"height": test_img_size, "width": test_img_size})
                                        ]).augment_image
                            ])


    test_dataset = AirbusDataset(ship_dir, ships_df, transform = test_tfs, mode="test")

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

        # assert(0)
    submission_df = pd.DataFrame(submission, columns=['ImageId', 'EncodedPixels'])

    res_df = submission_df.append(non_ships_df, ignore_index=True)
    
    res_df.to_csv('./out_submit/submission_v2_{}_{}.csv'.format(int(val_loss * 1000), test_img_size),
                                                                    index=False)
    res_df.to_csv('./submission_v2.csv', index=False)
    res_df.sample(10)
    print(val_loss ," / ", test_img_size)



