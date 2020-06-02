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
from model_UResnet34 import Resnet34Unet
from utils import *
from imgaug import augmenters as iaa
import imgaug

class AirbusDataset(Dataset):
    def __init__(self, data_dir, data_set, transform=None, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        self.data_set = data_set
        self.img_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                                std=(0.229, 0.224, 0.225))])
        self.transform = transform
        
    def  __getitem__(self ,index):
        image_name = self.data_set.iloc[index]["ImageId"]
        if self.mode == "train" or self.mode == "valid":
            imgpath = os.path.join(self.data_dir, "train_v2/"+image_name)
        else:
            imgpath = os.path.join(self.data_dir, "test_v2/"+image_name)

        imgO = Image.open(imgpath).convert('RGB')
        imgO = np.array(imgO)

        rle_pixels = []
        rle_pixels.append(self.data_set.iloc[index]["EncodedPixels"])
        

        mask = masks_as_image(rle_pixels)


        if self.transform is not None:
            imgO = self.transform(imgO)
            mask = self.transform(mask)


        imgO = self.img_transform(imgO)
        mask = torch.from_numpy(np.moveaxis(mask, -1, 0)).float()


        if self.mode == "train" or self.mode == "valid":
            return imgO,  mask
        else:
            return imgO, str(image_name)

    def __len__(self):
        return self.data_set.shape[0]

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        nums = y.size(0)
        smooth = 1.
        x = F.sigmoid(x)
        x_flat = x.view(nums, -1)
        y_flat = y.view(nums, -1)
        
        intersection = x_flat * y_flat
        loss = 2 * (intersection.sum(1) + smooth) / (x_flat.sum(1) + y_flat.sum(1) + smooth)
        loss = loss.sum() / nums
        
        return loss

class BCEDiceWithLogitsLoss(nn.Module):
    def __init__(self, dice_weight=1, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.smooth = smooth
        
    def __call__(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError("size mismatch, {} != {}".format(outputs.size(), targets.size()))
            
        loss = self.bce(outputs, targets)

        targets = (targets == 1.0).float()
        targets = targets.view(-1)
        outputs = F.sigmoid(outputs)
        outputs = outputs.view(-1)

        intersection = (outputs * targets).sum()
        dice = 2.0 * (intersection + self.smooth)  / (targets.sum() + outputs.sum() + self.smooth)
        
        loss -= self.dice_weight * torch.log(dice) # try with 1- dice

        return loss

class BCEJaccardWithLogitsLoss(nn.Module):
    def __init__(self, jaccard_weight=1, smooth=1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight
        self.smooth = smooth

    def forward(self, outputs, targets):
        if outputs.size() != targets.size():
            raise ValueError("size mismatch, {} != {}".format(outputs.size(), targets.size()))
            
        loss = self.bce(outputs, targets)

        if self.jaccard_weight:
            targets = (targets == 1.0).float()
            targets = targets.view(-1)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.view(-1)
           
            intersection = (targets * outputs).sum()
            union = outputs.sum() + targets.sum() - intersection

            loss -= self.jaccard_weight * torch.log((intersection + self.smooth ) / (union + self.smooth )) # try with 1-dice
        return loss


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

    
    # train_seq = iaa.Sequential([
    #                     iaa.Fliplr(0.5),
    #                     iaa.size.Resize({"height": 512, "width": 512})
    #                     ])

    train_tfs = transforms.Compose([ iaa.Sequential([
                                        iaa.Fliplr(0.5),
                                        iaa.size.Resize({"height": 512, "width": 512})
                                        ]).augment_image
                            ])

    train_dataset = AirbusDataset(ship_dir, train_df, transform = train_tfs)
    # assert(0)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=1)
    
    model = Resnet34Unet(3, 1)
    # Frozen parameters of the encoder layers
    child_counter = 0    
    for child in model.children():
        child_counter +=1
        for param in child.parameters():
            param.requires_grad = False
        if child_counter == 6:
            break

    model.to(device)
    diceloss = BCEDiceWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)


    if os.path.exists("./weights/uresnet34_best.pt"):
        print('Load pre-trained weights !! ')
        model.load_state_dict(torch.load("./weights/uresnet34_best.pt"))

    model.train()

    LOSS = 'BCEJaccardWithLogitsLoss'
    if LOSS == 'BCEWithDigits':
        criterion = nn.BCEWithLogitsLoss()
    elif LOSS == 'DiceLoss':
        criterion = DiceLoss()
    elif LOSS == 'BCEDiceWithLogitsLoss':
        criterion = BCEDiceWithLogitsLoss()
    elif LOSS == 'BCEJaccardWithLogitsLoss':
        criterion = BCEJaccardWithLogitsLoss()
    else:
        raise NameError("loss not supported")
    

    min_loss = np.inf
    step = 0
    for epoch in range(10):
        losses = 0
        for i, (imgO, labels) in enumerate(train_loader, 0):
            optim.zero_grad()
            imgO = imgO.to(device)       # [B, 3, H, W]
            labels = labels.to(device)   # [B, H, W] with class indices (0, 1)
            out = model(imgO)            # [B, 3, H, W]


            loss = criterion(out, labels)

            loss.backward()
            optim.step()

            losses += loss.item()
            print(loss)
            # assert(0)
        
        mean_loss = losses / len(train_loader)
        print('[%d] loss: %.3f' %  (epoch + 1, mean_loss))
        if losses < min_loss:
            torch.save(model.state_dict(), "./weights/uresnet34_best.pt")
            min_loss = mean_loss




    