```python
import torch
from torchvision import datasets, transforms

to_pil = transforms.ToPILImage()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/media/quazi/DATADRIVE1/data/cifar/', train=True, download=True,
        transform=transforms.ToTensor())
                     
for batch_idx, (data, _) in enumerate(train_loader):        
    img = to_pil(data[0].data.cpu())
    plt.imshow(img)
    plt.show()
```
