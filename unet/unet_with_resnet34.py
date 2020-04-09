from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
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
        # self.train_gp = self.train_gp.drop(self.train_gp.index[0:100000])
        to_remove = np.random.choice(self.train_gp[self.train_gp['exist_ship']==0].index,
                                            size=100000,replace=False)
        self.train_gp = self.train_gp.drop(to_remove)
        # print(self.train_gp["exist_ship"].value_counts())
        # print (self.train_gp.shape)
        self.train_sample = self.train_gp.sample(5000)
        # print(self.train_sample["exist_ship"].value_counts())
        # print(self.train_sample.iloc[0])
        # image_name = self.train_sample.iloc[0]["ImageId"]
        # print(image_name)
        # print(self.train_sample.shape[0])
    
    def  __getitem__(self ,index):
        image_name = self.train_sample.iloc[index]["ImageId"]
        label = self.train_sample.iloc[index]["exist_ship"]
        imgpath = os.path.join(self.data_dir, "train_v2/"+image_name)
        print(image_name)
        imgO = Image.open(imgpath).resize((256,256)).convert('RGB')
        imgO = np.array(imgO, dtype=np.float32)
        print(imgO.shape)
        if self.transform is not None :
            imgO = self.transform(imgO)
        print(imgO.size)
        return imgO, torch.Tensor(label)
    
    def __len__(self):
        return self.train_sample.shape[0]

def train_model(model, critertion, optimizer, scheduler, num_epochs=25, mode="train"):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if mode == "train":
            model.train()
        else:
            model.eval()
            
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase =="train":
                    loss.backward()
                    optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        #     if phase == "train":
        #         scheduler.step()

        #     epoch_loss = running_loss / dataset_sizes[phase]
        #     epoch_acc = running_corrects.double() / dataset_sizes[phase]

        #     print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        #         phase, epoch_loss, epoch_acc))

        #     # deep copy the model
        #     if phase == 'val' and epoch_acc > best_acc:
        #         best_acc = epoch_acc
        #         best_model_wts = copy.deepcopy(model.state_dict())

        # print()

    # time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # # load best model weights
    # model.load_state_dict(best_model_wts)
    return model

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

if __name__ == "__main__":
    csv_file = "train_ship_segmentations_v2.csv"
    data_dir = "./data_airbus"

    m = (0.485, 0.456, 0.406)
    s = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=m, std=s),
    ])

    airbus_dataset = AirbusDataset(csv_file, data_dir, transform = transform)
    # imgO, img, lbl = airbus_dataset[2]

    train_loader = DataLoader(dataset=airbus_dataset,
                            batch_size=2,
                            shuffle=False)
    for imgo , lbl in train_loader:
        # print(img.shape)
        print(imgo.shape)
    assert(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    resnet34 = models.resnet34(pretrained=True)
    for param in resnet34.parameters():
        param.requires_grad = True

    in_num_ftrs = resnet34.fc.in_features
    out_num_ftrs = resnet34.fc.out_features
    resnet34.fc = nn.Linear(in_num_ftrs, 2)

    
    resnet34 = resnet34.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_resnet = optim.SGD(resnet34.fc.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_resnet = train_model(resnet34, criterion, optimizer_resnet, exp_lr_scheduler, num_epochs=25)

    