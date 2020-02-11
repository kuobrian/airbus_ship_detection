## This kaggle competition airbus ship detection challenge

## what is EncodedPixels?

* It is the way to express the location of pixels containing an object, like a list in one dimension, instead of using two dimensions or coordenates.
If the object is on the row 1, between pixels 5 and 10, you can express it like : 5 5 , and like this for every row.

-----------
### Data Prepare

- [ ] Delete useless images without ships
- [ ] Find bad annotations and save them in the folder (../tmp/bad_anns);
- [ ] Transform RLE into COCO annotations
    + [pycoccreator](https://github.com/waspinator/pycococreator "Title")