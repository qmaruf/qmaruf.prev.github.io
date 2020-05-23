---
title: Matplotlib Subplots In Loop
date: 2018-11-23 00:00:00 Z
---

```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
axs = axs.ravel()
[axi.set_axis_off() for axi in axs.ravel()]

for i in range(100):
    img = np.random.rand(10, 10, 3)
    axs[i].imshow(img)
    axs[i].set_title(i)

plt.suptitle('Super Title')
plt.show()
```
