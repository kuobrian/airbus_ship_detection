## kaggle competition: airbus ship detection challenge

## what is EncodedPixels?

* It is the way to express the location of pixels containing an object, like a list in one dimension, instead of using two dimensions or coordenates.
If the object is on the row 1, between pixels 5 and 5, you can express it like : 5 5 , and like this for every row.

-----------
## Data Prepare

- [ ] Delete useless images without ships
- [ ] Add index for ships existence(0 or 1)
- [ ] drop and split train & valid set


## Detect ship existence - classification_ships_UResnet34.py
- [ ] Do transfer learning with Resnet34 
- [ ] Load training and valid set
- [ ] Image size 256 x 256, RGB data

## ship semantic segmentationn - train.py
- [ ] Use Unet Base on ResNet34 model 
- [ ] Load training and valid set
- [ ] Image size 256 x 256, RGB data


## Submittion result with predict ship + Unet34 - submit.py

Size| lr | train_loss | val_loss | IoU | dice | result |
:----:|:----:|:----:|:----:|:----:|:----:|:----:|
256x256 | 0.002 |    |    |    |    |    |
384x384 |    |    |    |    |    |      |
768x768 |    |    |    |    |    |    |

## Ref.

1. [Simple transfer Learning detect Ship exist (Keras)](https://www.kaggle.com/super13579/simple-transfer-learning-detect-ship-exist-keras "title")

2. [U-Net base on ResNet34 Transfer learning (Keras)](https://www.kaggle.com/super13579/u-net-base-on-resnet34-transfer-learning-keras/notebook "title")