```python
import torch
from torchvision import datasets, transforms

to_pil = transforms.ToPILImage()
def show_img_from_batch(batch):
    plt.figure()
    img = to_pil(batch[0].data.cpu())
    plt.imshow(img)
    
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('/media/quazi/DATADRIVE1/data/cifar/', train=True, download=True,
        transform=transforms.ToTensor())
                     
for batch_idx, (data, _) in enumerate(train_loader):        
    show_img_from_batch(data)
    
```
