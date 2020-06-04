import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

import torch.nn.functional as F
import torch.nn as nn
from model_UResnet34 import Resnet34Unet, double_conv
from utils import *
from classification_ships_UResnet34 import CustomResnet34
from imgaug import augmenters as iaa
import imgaug

from train import AirbusDataset, DiceLoss, MixedLoss, BCEJaccardWithLogitsLoss, BCEDiceWithLogitsLoss


# Get pre-trained classidication model
class CUResnet34(nn.Module):
    def __init__(self, in_channel, out_channel, training=True):
        super().__init__()
        self.training = training
        Uresnet34 = CustomResnet34(2)
        PATH = './weights/classification_ships_UResnet34.pt'
    
        if os.path.exists('./weights/classification_ships_UResnet34.pt'):
            print('Load pre-trained weights !! ')
            checkpoint = torch.load('./weights/classification_ships_UResnet34.pt')
            Uresnet34.load_state_dict(checkpoint['weights'])
            dev_acc = checkpoint['dev_acc']
            dev_loss = checkpoint['dev_loss']

        resnet34 = next(Uresnet34.children())

        self.conv1 = nn.Sequential(*list(resnet34.children())[:3])
        self.pool1 = resnet34.maxpool
        self.conv2_down = nn.Sequential(*list(resnet34.children())[4])
        self.conv3_down = nn.Sequential(*list(resnet34.children())[5])
        self.conv4_down = nn.Sequential(*list(resnet34.children())[6])
        self.conv5_down = nn.Sequential(*list(resnet34.children())[7])
        

        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_up = double_conv(256 + 512, 256)
        self.conv2_up = double_conv(128 + 256, 128)
        self.conv3_up = double_conv(64 + 128, 64)

        self.dconv_last = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, out_channel, 1)
        )

    def forward(self, x):
        out_conv1 = self.conv1(x)
        temp = self.pool1(out_conv1)
        out_conv2 = self.conv2_down(temp)
        out_conv3 = self.conv3_down(out_conv2)
        out_conv4 = self.conv4_down(out_conv3)
        bottle = self.conv5_down(out_conv4)

        up_x = self.upsample(bottle)
        up_x = torch.cat([up_x, out_conv4], dim=1)
        up_x = self.conv1_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv3], dim=1)
        up_x = self.conv2_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv2], dim=1)
        up_x = self.conv3_up(up_x)

        up_x = self.upsample(up_x)
        up_x = torch.cat([up_x, out_conv1], dim=1)
        out = self.dconv_last(up_x)

        return out


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    
    train_tfs = transforms.Compose([ iaa.Sequential([
                                        iaa.Fliplr(0.5),
                                        iaa.size.Resize({"height": 256, "width": 256})
                                        ]).augment_image
                            ])

    train_dataset = AirbusDataset(ship_dir, train_df, transform = train_tfs)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=1)


    val_tfs = transforms.Compose([ iaa.Sequential([
                                        iaa.size.Resize({"height": 256, "width": 256})
                                        ]).augment_image
                            ])
    valid_dataset = AirbusDataset(ship_dir, valid_df, transform = val_tfs)
    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=64,
                            shuffle=True,
                            num_workers=1)
    model = CUResnet34(3, 1).to(device) 
    model.eval()

    # Frozen parameters of the encoder layers
    child_counter = 0    
    for child in model.children():
        child_counter +=1
        for param in child.parameters():
            param.requires_grad = False
        if child_counter == 6:
            break

    LOSS = 'MixedLoss'
    optim = torch.optim.Adam(model.parameters(), lr=0.002)
 
    if LOSS == 'BCEWithDigits':
        criterion = nn.BCEWithLogitsLoss()
    elif LOSS == 'DiceLoss':
        criterion = DiceLoss()
    elif LOSS == 'BCEDiceWithLogitsLoss':
        criterion = BCEDiceWithLogitsLoss()
    elif LOSS == 'BCEJaccardWithLogitsLoss':
        criterion = BCEJaccardWithLogitsLoss()
    elif LOSS == "MixedLoss":
        criterion = MixedLoss(10.0, 2.0)
    else:
        raise NameError("loss not supported")

    min_loss = np.inf
    Iou = 0
    step = 0

    if os.path.exists("./weights/CUResnet34_best.pt"):
        print('Load pre-trained weights !! ')
        checkpoint = torch.load("./weights/CUResnet34_best.pt")
        model.load_state_dict(checkpoint['weights'])
        Iou = checkpoint['Iou']
        min_loss = checkpoint['dev_loss']
    print(min_loss, Iou)

    for epoch in range(10):
        losses = 0
        for batch_idx, (imgO, labels) in enumerate(train_loader, 0):
            model.train()
            optim.zero_grad()
            imgO = imgO.to(device)       # [B, 3, H, W]
            labels = labels.to(device)   # [B, H, W] with class indices (0, 1)
            out = model(imgO)            # [B, 3, H, W]

            loss = criterion(out, labels)
            loss.backward()
            optim.step()
            losses += loss.item()
            step += 1
            if step % 100 == 0:
                print('\t Step: {}, loss: {}'.format(step, losses/(batch_idx + 1) ))


        model.eval()
        dev_loss, IoU_value, diceloss = 0, 0, 0
        with torch.no_grad():
            for dev_batch_idx, (val_imgs, val_masks) in enumerate(valid_loader):
                val_imgs = val_imgs.to(device)
                val_masks = val_masks.to(device)
                val_outputs = model(val_imgs)

                dev_loss += criterion(val_outputs, val_masks).item()
                IoU_value += IoU(val_outputs, val_masks).item()
                diceloss += DiceLoss().forward(val_outputs, val_masks).item()

        mean_loss = losses / len(train_loader)
        print('[{}] loss: {}, val loss: {} , IOU: {}, dice loss: {}'.format(epoch + 1,
                                                                            mean_loss,
                                                                            dev_loss/len(valid_loader),
                                                                            IoU_value/len(valid_loader),
                                                                            diceloss/len(valid_loader)))
        if (dev_loss/len(valid_loader)) < min_loss:
            print("Save model weights")
            torch.save(
                {"weights":model.state_dict(),
                    "Iou": IoU_value/len(valid_loader),
                    "dev_loss": dev_loss/len(valid_loader)}, "./weights/CUResnet34_best.pt")
            min_loss = dev_loss/len(valid_loader)
