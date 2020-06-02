import os
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.morphology import label
import pandas as pd
import matplotlib.pyplot as plt

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

def unique_images(data_dir, train_path, test_path):
    masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))

    ''' Test encode decode image and RLE '''
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
    # rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']
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
    # plt.show()
    masks.drop(['ships'], axis=1, inplace=True)

    return unique_img_ids, masks


def balanced_images(masks, unique_img_ids, test_size=0.05):
    if not os.path.exists("./processing_data/balanced_train_set.csv") or\
        not os.path.exists("./processing_data/balanced_valid_set.csv"):

        train_ids, valid_ids = train_test_split(unique_img_ids, 
                                            test_size = test_size, 
                                            stratify = unique_img_ids['ships'])
        print(train_ids.head())
        train_df = pd.merge(masks, train_ids)
        valid_df = pd.merge(masks, valid_ids)

        train_df['ships'].hist(bins=np.arange(10))
        plt.show()

        train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x+1)//2).clip(0, 7)

        balanced_train_df = train_df.groupby('grouped_ship_count').apply(sample_ships)
        
        balanced_train_df['ships'].hist(bins=np.arange(10))
        plt.show()

        balanced_train_df.to_csv("./processing_data/balanced_train_set.csv")
        valid_df.to_csv("./processing_data/balanced_valid_set.csv")
    else:
        balanced_train_df = pd.read_csv("./processing_data/balanced_train_set.csv")
        valid_df = pd.read_csv("./processing_data/balanced_valid_set.csv")

    return balanced_train_df, valid_df



def drop_img_without_ships(masks, unique_img_ids, test_size=0.05):
    if not os.path.exists("./processing_data/droped_train_set.csv") or\
        not os.path.exists("./processing_data/droped_valid_set.csv"):

        unique_img_ids = unique_img_ids[unique_img_ids.ships != 0]
        train_ids, valid_ids = train_test_split(unique_img_ids, 
                                            test_size = test_size, 
                                            stratify = unique_img_ids['ships'])

        # train_ids['ships'].hist(bins=np.arange(10))
        # plt.show()
        train_ids = pd.merge(masks, train_ids)
        valid_ids = pd.merge(masks, valid_ids)

        train_ids.to_csv("./processing_data/droped_train_set.csv")
        valid_ids.to_csv("./processing_data/droped_valid_set.csv")

    else:
        train_ids = pd.read_csv("./processing_data/droped_train_set.csv")
        valid_ids = pd.read_csv("./processing_data/droped_valid_set.csv")

    return train_ids, valid_ids





# if __name__ == "__main__":
#     ship_dir = "./data"
#     train_path = os.path.join(ship_dir, "train_v2")
#     test_path = os.path.join(ship_dir, "test_v2")


#     if not os.path.exists("./processing_data/unique.csv"):
#         unique_img_ids, masks = unique_images(ship_dir, train_path, test_path)
#         unique_img_ids.to_csv("./processing_data/unique.csv")
#     else:
#         unique_img_ids = pd.read_csv("./processing_data/unique.csv")
#         masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))

#     # balanced training data
#     train_df, valid_df = balanced_images(masks, unique_img_ids)

#     # drop images without ships
#     train_df, valid_df = drop_img_without_ships(masks, unique_img_ids)
