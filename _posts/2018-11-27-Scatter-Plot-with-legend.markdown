---
title: Scatter Plot With Legend
date: 2018-11-27 00:00:00 Z
---

This code will create random features and their 2D embedding to plot using a matplotlib scatter plot. Just keeping it here for future reference. `colorbar` is used as legend.

```python

import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
%matplotlib inline

X, y = [], []
for i in range(1000):
  feats = np.random.rand(100)
  X.append(feats)
  y.append(i%10)
  
tsne = manifold.TSNE(n_components=2)
latent = tsne.fit_transform(X)

plt.scatter(latent[:, 0], latent[:, 1], c=y, label=y)
plt.colorbar()
plt.grid(True)

```
