```
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from glob import glob
import cv2

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

paths = glob('/media/quazi/DATADRIVE1/data/cifar/train/airplane/train_imgs/imgs/*.png')

imgs = [img_to_array(load_img(path)).reshape(1, 32, 32, 3) for path in paths]
print (imgs[0].shape)

dirp = '/media/quazi/DATADRIVE1/data/cifar/train/airplane/train_imgs/preview'
for x in imgs:
    i = 0
    for batch in datagen.flow(imgs,save_to_dir=dirp, save_prefix='augmented', save_format='jpeg'):
        i += 1
        if i > 50:
            break 
```