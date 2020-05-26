1.  how to prepare input:
===========================

image = <... read image data from from parquet file ... >.reshape(137, 236)
image = 1-image.astype(np.float32)/255



def to_64x112(image):
    image = cv2.resize(image,dsize=None, fx=64/137,fy=64/137,interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image,0,0,1,1,cv2.BORDER_CONSTANT,0)
    return image





2.  model.py, seresnext.py
===========================
modified se-resnext50, see also

see also resnet50-D of [1]:

[1] "Bag of Tricks for Image Classification with Convolutional Neural Networks"
- Tong He, cvpr 2019




3.  augmentation.py
===========================
list of augmentations that works



4.  train_rand_aug2c.py
===========================
reference training code to produce training log: 
- ignore the "iter" and "epoch" values, you may not need to train that long
- instead, take note of the learning rate, batch size and the expected local validation values of
  kaggle metrics you should get


(you may need to adjust the augmentation hyperparameter magnitude, the given values in train_rand_aug2c.py
 are not "fully optimised" )
