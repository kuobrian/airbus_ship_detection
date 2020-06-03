import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import time
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def PreProcess(data_dir):
    #corrupted image
    exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg',  '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']
    
    train_names = [f for f in os.listdir(os.path.join(data_dir, "train_v2"))]
    test_names = [f for f in os.listdir(os.path.join(data_dir, "test_v2"))]
    

    for name in exclude_list:
        if(name in train_names): 
            train_names.remove(name)
        if(name in test_names): 
            test_names.remove(name)


    masks = pd.read_csv(os.path.join(data_dir, 'train_ship_segmentations_v2.csv'))
    test_masks = pd.read_csv(os.path.join(data_dir, "sample_submission_v2.csv"))


    masks["ships"] = masks["EncodedPixels"].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

    masks = masks[masks["ImageId"].isin(train_names)]
    test_df = test_masks[test_masks["ImageId"].isin(test_names)]

    # masks["ships"].hist()
    # plt.show()

    # 5% of data in the validation set is sufficient for model evaluation
    tr_df, val_df = train_test_split(masks, test_size=0.05, random_state=42)

    return tr_df, val_df, test_df


class AirbusDataset(Dataset):
    def __init__(self, data_dir, data_set, transform = None):
        self.data_set = data_set
        self.data_dir = data_dir
        self.transform = transform
        
    def  __getitem__(self ,index):
        image_name = self.data_set.iloc[index]["ImageId"]
        label = self.data_set.iloc[index]["ships"]
        imgpath = os.path.join(self.data_dir, "train_v2/"+image_name)
        imgO = Image.open(imgpath).convert('RGB')
        if self.transform is not None :
            imgO = self.transform(imgO)
        
        return imgO,  label
    
    def __len__(self):
        return self.data_set.shape[0]

def pre_process_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train_ship_segmentations_v2.csv"))
    print(train.head())

    ''' Tranfer EncodedPixels to target '''
    train["exist_ship"] = train['EncodedPixels'].fillna(0)
    train.loc[train["exist_ship"]!=0,"exist_ship"]=1
    
    ''' duplicate image '''
    print(len(train['ImageId']))
    print(train['ImageId'].value_counts().shape[0])
    train_gp = train.groupby('ImageId').sum().reset_index()
    train_gp.loc[train_gp['exist_ship']>0,'exist_ship']=1


    ''' Balance have chip and no chip data '''
    print(train_gp['exist_ship'].value_counts())
    train_gp = train_gp.sort_values(by='exist_ship')
    train_gp = train_gp.drop(train_gp.index[0:100000])

    ''' Set training set count '''
    print(train_gp['exist_ship'].value_counts())
    train_sample = train_gp.sample(5000)
    print(train_sample['exist_ship'].value_counts())
    print (train_sample.shape)

    ''' load training data function '''
    train_path = os.path.join(data_dir, "train_v2")
    test_path = os.path.join(data_dir, "test_v2")

class CustomResnet34(nn.Module):
    def __init__(self, num_classes, training = True):
        super().__init__()
        self.training = training
        self.num_classes = num_classes
        self.resnet = models.resnet34(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.in_num_ftrs = self.resnet.fc.in_features
        self.out_num_ftrs = self.resnet.fc.out_features

        self.fc1 = nn.Linear(self.out_num_ftrs, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    csv_file = "train_ship_segmentations_v2.csv"
    data_dir = "./data"
    print(device)

    train_df, val_df, test_df = PreProcess(data_dir)


    m = (0.485, 0.456, 0.406)
    s = (0.229, 0.224, 0.225)
    tfs = transforms.Compose([
        transforms.Resize((256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s)
    ])

    train_dataset = AirbusDataset(data_dir, train_df, transform = tfs)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=4)

    val_dataset = AirbusDataset(data_dir, val_df, transform = tfs)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=64,
                            shuffle=False,
                            num_workers=4)


    y_onehot = torch.FloatTensor(2, 2)
   
    # resnet34 = models.resnet34(pretrained=True)
    Uresnet34 = CustomResnet34(2)
    # for param in resnet34.parameters():
    #     print(param.requires_grad)

    Uresnet34 = Uresnet34.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_resnet = optim.SGD(Uresnet34.parameters(), lr=0.002)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_resnet, step_size=7, gamma=0.1)

    PATH = './weights/classification_ships_UResnet34.pt'
    min_loss = np.inf
    step = 0
    for epoch in range(10):
        running_loss = 0.0
        for batch_idx, (imgO, labels) in enumerate(train_loader):
            # y_onehot.zero_()
            # y_onehot.scatter_(1, labels, 1)
            # print(imgO.shape)
            # print(y_onehot.shape)
            Uresnet34.train()
            imgO = imgO.to(device)
            labels = labels.to(device)
            outputs = Uresnet34(imgO)
            
            optimizer_resnet.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_resnet.step()
            running_loss += loss.item()
            step += 1
            if step % 10 == 0:
                print('\t Step: {}, loss: {}'.format(step, running_loss/(batch_idx + 1) ))

        # if (epoch + 1) % 5 == 0 :
        Uresnet34.eval()
        n_dev_correct, dev_loss = 0, 0
        with torch.no_grad():
            for dev_batch_idx, (val_imgs, val_labels) in enumerate(val_loader):
                val_outputs = Uresnet34(val_imgs)
                n_dev_correct += (torch.max(val_outputs, 1)[1].view(val_labels.size()) == val_labels).sum().item()
                dev_loss = criterion(val_outputs, val_labels)
        
        dev_acc = 100. * n_dev_correct / len(val_loader)
        running_loss = running_loss/len(train_loader)
        print('[{}] loss: {}, val acc: {}, val loss: {}'.format(epoch + 1, running_loss, dev_acc, dev_loss))
        if min_loss > running_loss:
            torch.save({"weights": resnet34.state_dict(),
                        "dev_acc": dev_acc,
                        "dev_loss": dev_loss}, PATH)
            min_loss = running_loss
            
    

    