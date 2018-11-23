```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
axs = axs.ravel()
[axi.set_axis_off() for axi in axs.ravel()]

for i in range(100):
    img = np.random.rand(10, 10, 3)
    axs[i].imshow(img)
    axs[i].set_title(i)

```
