## This kaggle competition airbus ship detection challenge

## what is EncodedPixels?

* It is the way to express the location of pixels containing an object, like a list in one dimension, instead of using two dimensions or coordenates.
If the object is on the row 1, between pixels 5 and 5, you can express it like : 5 5 , and like this for every row.

-----------
## Data Prepare

- [ ] Delete useless images without ships
- [ ] Find bad annotations and save them in the folder (../tmp/bad_anns);
- [ ] Transform RLE into COCO annotations
    + [pycoccreator](https://github.com/waspinator/pycococreator "Title")



## Detect ship exist or not - unet_with_resnet34.py
- [ ] Use Resnet34 to do transfer learning
- [ ] Load 5000 images to be training
- [ ] Split ~4000 training set, ~1000 validate set
- [ ] Image size 256 x 256, RGB data


## Submittion result with predict ship + Unet34 
- [ ] Use Unet Base on ResNet34 model
- [ ] Load 5000 images to be training
- [ ] Use bce_log+dice_loss for ResNet34+Unet loss function
- [ ] Image size 256 x 256, RGB data

## Ref.

[Simple transfer Learning detect Ship exist (Keras)](https://www.kaggle.com/super13579/simple-transfer-learning-detect-ship-exist-keras "title")
[U-Net base on ResNet34 Transfer learning (Keras)](https://www.kaggle.com/super13579/u-net-base-on-resnet34-transfer-learning-keras/notebook "title")