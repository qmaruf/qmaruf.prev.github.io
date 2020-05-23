---
title: How To Use Keras And Vgg For Object Classification
date: 2018-07-25 00:00:00 Z
---

In this post, i'll show an example of how to use pretrained vgg network for object classification using keras.
The first part is to import necessary library.

```
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras_applications.vgg16 import preprocess_input, decode_predictions

```
Let's get our model.
```
model = VGG16()
```

https://upload.wikimedia.org/wikipedia/commons/3/32/House_sparrow04.jpg



layer_name = 'fc1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(img)
