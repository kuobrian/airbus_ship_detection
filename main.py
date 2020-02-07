import os
import pandas as pd
import numpy as np
from skimage.data import imread
import matplotlib.pyplot as plt
from PIL import Image
import random



'''TODO
1. Delete useless images without ships;
2. Find bad annotations and save them in the folder (../tmp/bad_anns);
3. Transform RLE into COCO annotations;
'''

# Delete useless images without ships
def Useful_Traing_set(dataset_train, train_csv):
    # read csv
    df = pd.read_csv(train_csv)
    print("Dataframe lines: ", df.shape[0])

    df = df.dropna(axis=0)
    num_of_ships = df.shape[0]
    print("Inastances      ： ",num_of_ships)

    images = set()
    for i in range(num_of_ships):
        if df.iloc[i, 0] not in images:
            images.add(df.iloc[i, 0])
    print("Images with ship: ", len(images))

    train_valid_file = "./data/train_valid.txt"
    count = 0
    imgs_path = os.listdir(dataset_train)
    with open(train_valid_file, "w") as out_file:
        for im in imgs_path:
            if im in images:
                out_file.write(os.path.join(dataset_train, im) + "\n")
                count += 1
    print("Useful train image file name in train_valid.txt (%d images)" % count)
    return df

def



if __name__ == "__main__":
    dataset_train = "./data/train_v2"
    train_csv = "./data/train_ship_segmentations_v2.csv"
    df = Useful_Traing_set(dataset_train, train_csv)
