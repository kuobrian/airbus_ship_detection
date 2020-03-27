import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import math
import json
import re
import datetime
import fnmatch
from pycococreator import pycococreator
# from skimage.data import imread

'''TODO
1. Delete useless images without ships;
2. Find bad annotations and save them in the folder (../tmp/bad_anns);
3. Transform RLE into COCO annotations;
'''

# Delete useless images without ships
def Useful_Traing_set(dataset_train, train_csv):
    # read csv
    df = pd.read_csv(train_csv)
    print("num of data: ", df.shape[0])

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

# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts =  np.asarray(s[0::2], dtype=int)
    lengths = np.asarray(s[1::2], dtype=int)
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def csv_show_rle(ImageId, dataset, df):
    img_path = os.path.join(dataset, ImageId)
    img = Image.open(img_path)
    rle_masks = df.loc[df["ImageId"] == "000155de5.jpg", "EncodedPixels"].tolist()
    
    if isinstance(rle_masks[0], str):
        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768))
        for mask in rle_masks:
            binary_mask = rle_decode(mask)
            print('Area: ', np.sum(binary_mask))
            all_masks += binary_mask

        fig, axarr = plt.subplots(1, 3)
        axarr[0].axis('off'),
        axarr[1].axis('off'),
        axarr[2].axis('off')
        axarr[0].imshow(img),
        axarr[1].imshow(all_masks),
        axarr[2].imshow(img)
        axarr[2].imshow(all_masks, alpha=0.4)
        plt.tight_layout(h_pad=0.1, w_pad=0.1)
        plt.show()









INFO = {
    "description": "Kaggle Dataset",
    "url": "https://github.com/kuobrian/",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "kuobrian",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'ship',
        'supercategory': 'ship',
    },
]


def save_bad_ann(img_path, image_name, mask, segmentation_id):
    img = Image.open(img_path)
    fig, axarr = plt.subplots(1, 3)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[2].axis('off')
    axarr[0].imshow(img)
    axarr[1].imshow(mask)
    axarr[2].imshow(img)
    axarr[2].imshow(mask, alpha=0.4)
    plt.tight_layout(h_pad=0.1, w_pad=0.1)
    if not os.path.exists("./tmp"):
    	os.makedirs("./tmp")
    plt.savefig( os.path.join("./tmp", image_name.split(".")[0] +"_" +str(segmentation_id) +".png") )
    plt.close()


def turn2coco(images_path, df):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    num_of_image_files = len(images_path)

    for img_path in images_path:
        image = Image.open(img_path)
        iname = os.path.basename(img_path)
        image_info = pycococreator.create_image_info(image_id, iname, image.size)
        coco_output["images"].append(image_info)

        rle_masks = df.loc[df['ImageId'] == iname, 'EncodedPixels'].tolist()
        num_of_rle_masks = len(rle_masks)

        for index in range(num_of_rle_masks):
            binary_mask = rle_decode(rle_masks[index])
            class_id = 1
            category_info = {"id": class_id, "is_crowd": 0}
            annotation_info = pycococreator.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, tolerance=2)
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
            else:
                save_bad_ann(img_path, iname, binary_mask, segmentation_id)
            segmentation_id = segmentation_id + 1

        print("%d of %d is done."%(image_id, num_of_image_files))
        image_id = image_id + 1

    
    with open('./data/annotations/instances_ships_train2018.json', 'w') as output_json_file:
        # json.dump(coco_output, output_json_file)
        json.dump(coco_output, output_json_file, indent=4)



if __name__ == "__main__":
    dataset_train = "./data/train_v2"
    dataset_test = "./data/test_v2"
    train_csv = "./data/train_ship_segmentations_v2.csv"
    test_csv = "./data/sample_submission_v2.csv"

    df = Useful_Traing_set(dataset_train, train_csv)
    
    if not os.path.exists("./data/train_valid.txt"):
        df = Useful_Traing_set(dataset_train, train_csv)

    with open("./data/train_valid.txt", "r") as file:
        lines = file.readlines()

    images_path = [x.strip() for x in lines]
    
    df = pd.read_csv(train_csv)
    
    count = 0
    ImageId_list = os.listdir(dataset_train)
    ImageId = random.choice(ImageId_list)
    csv_show_rle(ImageId, dataset_train, df)
    if not os.path.exists("./data/annotations/instances_ships_train2018.json"):
        turn2coco(images_path, df)

