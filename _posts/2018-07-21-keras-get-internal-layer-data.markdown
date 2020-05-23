---
title: Keras Get Internal Layer Data
date: 2018-07-21 00:00:00 Z
---


```python
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.models import Model
%matplotlib inline
```

```python
model = VGG16()
image = load_img('./cat.jpg', target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
pred = model.predict(image)
pred = decode_predictions(pred)
print (pred)
```

    [[('n02124075', 'Egyptian_cat', 0.8634043), ('n02441942', 'weasel', 0.042830825), ('n02123045', 'tabby', 0.024431724), ('n02443484', 'black-footed_ferret', 0.013195711), ('n02123159', 'tiger_cat', 0.010636494)]]



```python
plot_model(model, 'vgg16.jpg')
plt.figure(figsize=(100, 100))
plt.imshow(plt.imread('./vgg16.jpg'))
```




    <matplotlib.image.AxesImage at 0x7fbb81485a20>




![png](output_2_1.png)



```python
layer_name = 'fc2'
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(image)
```


```python
print(intermediate_output)
```

    [[0.        1.2463692 1.9213017 ... 0.        1.0107884 0.6229787]]

