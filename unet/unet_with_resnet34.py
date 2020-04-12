from __future__ import print_function, division

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
import copy
import os
import gc
import pandas as pd
from PIL import Image

class AirbusDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.train_csv = pd.read_csv(os.path.join(data_dir, csv_file))
        self.data_dir = data_dir
        self.transform = transform
        self.valid_trainset = self.PreProcess()

    def PreProcess(self):
        if not os.path.exists("./exist_ships.csv"):
            # ''' delete annotations without ship '''
            # self.train_csv = self.train_csv.dropna(axis=0)
            num_of_ships = self.train_csv.shape[0]

            ''' Add exist_ship labels'''
            self.train_csv["exist_ship"] = self.train_csv['EncodedPixels'].fillna(0)
            self.train_csv.loc[self.train_csv["exist_ship"] != 0, "exist_ship"] = 1
            del self.train_csv["EncodedPixels"]

            ''' duplicate image '''
            print(len(self.train_csv["ImageId"]))
            print(self.train_csv["ImageId"].value_counts().shape[0])
            self.train_gp  = self.train_csv.groupby("ImageId").sum().reset_index()
            self.train_gp .loc[self.train_gp ["exist_ship"]>0, "exist_ship"] = 1
            
            print(self.train_gp["exist_ship"].value_counts())
            self.train_gp.to_csv("./exist_ships.csv")
        else:
            self.train_gp = pd.read_csv("./exist_ships.csv")

        self.train_gp= self.train_gp.sort_values(by="exist_ship")


        self.train_gp = self.train_gp.drop(self.train_gp.index[0:100000])
        to_remove = np.random.choice(self.train_gp[self.train_gp['exist_ship']==0].index,
                                            size=100000, replace=True)

        self.train_gp = self.train_gp.drop(to_remove)
        self.train_sample = self.train_gp.sample(5000)
        
    def  __getitem__(self ,index):
        image_name = self.train_sample.iloc[index]["ImageId"]
        label = self.train_sample.iloc[index]["exist_ship"]
        imgpath = os.path.join(self.data_dir, "train_v2/"+image_name)
        imgO = Image.open(imgpath).convert('RGB')
        if self.transform is not None :
            imgO = self.transform(imgO)
        
        return imgO,  label
    
    def __len__(self):
        return self.train_sample.shape[0]

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
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
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
    data_dir = "./data_airbus"
    
    m = (0.485, 0.456, 0.406)
    s = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize((256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s)
    ])

    airbus_dataset = AirbusDataset(csv_file, data_dir, transform = transform)
    # imgO, img, lbl = airbus_dataset[2]

    train_loader = DataLoader(dataset=airbus_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=1)
    y_onehot = torch.FloatTensor(2, 2)


    # resnet34 = models.resnet34(pretrained=True)


    resnet34 = CustomResnet34(2)
    # for param in resnet34.parameters():
    #     print(param.requires_grad)

    resnet34 = resnet34.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_resnet = optim.SGD(resnet34.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_resnet, step_size=7, gamma=0.1)


    for epoch in range(2):
        running_loss = 0.0
        for i, (imgO, labels) in enumerate(train_loader, 0):

            # y_onehot.zero_()
            # y_onehot.scatter_(1, labels, 1)
            # print(imgO.shape)
            # print(y_onehot.shape)
            # pass 

            imgO = imgO.to(device)
            labels = labels.to(device)
            optimizer_resnet.zero_grad()

            outputs = resnet34(imgO)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_resnet.step()


        print('[%d] loss: %.3f' %  (epoch + 1, running_loss))
    

    