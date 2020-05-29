import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
from resnet34Unet import Resnet34Unet


def rle_encode(image):
    pixels = image.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask, shape=(768, 768)):
    s = mask.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def masks_as_image(in_mask_list):
    all_masks = np.zeros((768, 768), dtype=np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def multi_rle_encode(image):
    labels = label(image[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def sample_ships(group_df, base_rep_val=1500):
    if group_df['ships'].values[0]==0:
        return group_df.sample(base_rep_val//3) # even more strongly undersample no ships
    else:
        return group_df.sample(base_rep_val, replace=(group_df.shape[0]<base_rep_val))

def preprocess(data_dir, train_path, test_path):
    masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))

    ''' Test encode decode image and RLE '''
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
    rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']
    print(type(rle_0))
    # image0 = masks_as_image(rle_0)
    # ax1.imshow(image0[:, :, 0])
    # ax1.set_title('Image$_0$')

    # # encode image0 and decode
    # rle_1 = multi_rle_encode(image0)
    # image1 = masks_as_image(rle_1)
    # ax2.imshow(image1[:, :, 0])
    # ax2.set_title('Image$_1$')

    masks["ships"] = masks["EncodedPixels"].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
    

    unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
    unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x > 0 else 0.0)
    unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
    unique_img_ids['file_size_kb'] = unique_img_ids['ImageId'].map(lambda c_img_id: 
                                    os.stat(os.path.join(train_path, c_img_id)).st_size/1024)
    # keep only 50kb files
    unique_img_ids = unique_img_ids[unique_img_ids['file_size_kb']>50]
    # unique_img_ids['file_size_kb'].hist()
    masks.drop(['ships'], axis=1, inplace=True)
    train_ids, valid_ids = train_test_split(unique_img_ids, 
                test_size = 0.3, 
                stratify = unique_img_ids['ships'])

    
    train_df = pd.merge(masks, train_ids)
    valid_df = pd.merge(masks, valid_ids)

    train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x+1)//2).clip(0, 7)

    balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)
    
    balanced_train_df['ships'].hist(bins=np.arange(10))

    balanced_train_df.to_csv("./train_set.csv")
    valid_df.to_csv("./valid_set.csv")

    return balanced_train_df, valid_df

class AirbusDataset(Dataset):
    def __init__(self, data_dir, train_set, img_transform=None, mask_transform=None):
        self.data_dir = data_dir

        self.train_set = train_set

        self.img_transform = img_transform
        self.mask_transform = mask_transform
        
    def  __getitem__(self ,index):
        image_name = self.train_set.iloc[index]["ImageId"]
        imgpath = os.path.join(self.data_dir, "train_v2/"+image_name)
        imgO = Image.open(imgpath).convert('RGB')

        if self.img_transform is not None :
            imgO = self.img_transform(imgO)

        rle_pixels = []
        rle_pixels.append(self.train_set.iloc[index]["EncodedPixels"])
        
        mask = masks_as_image(rle_pixels)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return imgO,  mask.float()
    
    def __len__(self):
        return self.train_set.shape[0]


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


def compute_metrics(pred, true, batch_size=16, threshold=0.5):
    pred = pred.view(batch_size, -1)
    true = true.view(batch_size, -1)
    
    pred = (pred > threshold).float()
    true = (true > threshold).float()
    
    pred_sum = pred.sum(-1)
    true_sum = true.sum(-1)
    
    neg_index = torch.nonzero(true_sum == 0)
    pos_index = torch.nonzero(true_sum >= 1)
    
    dice_neg = (pred_sum == 0).float()
    dice_pos = 2 * ((pred * true).sum(-1)) / ((pred + true).sum(-1))
    
    dice_neg = dice_neg[neg_index]
    dice_pos = dice_pos[pos_index]
    
    dice = torch.cat([dice_pos, dice_neg])
    jaccard = dice / (2 - dice)
    
    return dice, jaccard
    
class metrics:
    def __init__(self, batch_size=16, threshold=0.5):
        self.threshold = threshold
        self.batchsize = batch_size
        self.dice = []
        self.jaccard = []
    def collect(self, pred, true):
        pred = torch.sigmoid(pred)
        dice, jaccard = compute_metrics(pred, true, batch_size=self.batchsize, threshold=self.threshold)
        self.dice.extend(dice)
        self.jaccard.extend(jaccard)
    def get(self):
        dice = np.nanmean(self.dice)
        jaccard = np.nanmean(self.jaccard)
        return dice, jaccard


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ship_dir = "../data"
    train_path = os.path.join(ship_dir, "train_v2")
    test_path = os.path.join(ship_dir, "test_v2")

    if not os.path.exists("./train_set.csv") and not os.path.exists("./valid_set.csv"):
        train_df, valid_df = preprocess(ship_dir, train_path, test_path)
    else:
        train_df = pd.read_csv("./train_set.csv")
        valid_df = pd.read_csv("./valid_set.csv")

    m = (0.485, 0.456, 0.406)
    s = (0.229, 0.224, 0.225)
    img_transform = transforms.Compose([
        transforms.Resize((512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s)
    ])

    mask_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(512),
        transforms.ToTensor()])

    val_transform = transforms.Compose([
        transforms.Resize((512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s)
    ])


    train_dataset = AirbusDataset(ship_dir, train_df,\
                                img_transform=img_transform, mask_transform=mask_transform)


    valid_dataset = AirbusDataset(ship_dir, valid_df,\
                                img_transform=val_transform, mask_transform=mask_transform)

    train_loader = DataLoader(dataset=train_dataset,
                        batch_size=16,
                        shuffle=True,
                        num_workers=1)
    
    tvalid_loader = DataLoader(dataset=valid_dataset,
                        batch_size=4,
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
    for epoch in range(1):
        losses = 0
        # valid_metrics = metrics(batch_size=4)  # for validation


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
        

        print('[%d] loss: %.3f' %  (epoch + 1, losses))
        if losses < min_loss:
            torch.save(model.state_dict(), "./uresnet34_best.pt")
            min_loss = running_loss




    