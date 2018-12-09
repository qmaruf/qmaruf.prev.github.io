This simple pytorch class will load images from a folder. It'll not assign any label with the images. 

```python
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from glob import glob
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

class MyCustomDataset(Dataset):
    def __init__(self, images_path):
        self.images = [plt.imread(image_path) for image_path in tqdm(images_path)]
        self.transforms = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        image = self.images[index]
        image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.images)

imgs = glob('~/images/*ppm')
custom_dataset = MyCustomDataset(imgs)
data_loader = DataLoader(custom_dataset, batch_size=16, shuffle=True, num_workers=4)

for data in data_loader:
  pass
  # your magic goes here.
```
