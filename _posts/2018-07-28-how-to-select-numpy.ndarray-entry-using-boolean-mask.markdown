Well, we have collected cifar10 dataset from keras and now we want to select only those images which are from class 1. That means we want to select some entries from a 4 dimensional numpy ndarray `(x_train)`using a mask and we want to apply that mask in dimension `0` of `x_train`. The little code snippet below will serve this purpose. 
```
from keras.datasets import cifar10
(x_train, y_train), (_, _) = cifar10.load_data()
mask = np.squeeze(y_train == 1)
x_train_1 = x_train[mask, :, :, :]
print(x_train_1.shape)
```