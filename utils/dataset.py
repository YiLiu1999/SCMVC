import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class HSIDataset(Dataset):
    """Hyperspectral Image Dataset for multiview contrastive learning.

    Creates three views from each sample:
    - Spatial view 1: RGB bands [0:3] with augmentation
    - Spatial view 2: RGB bands [3:6] with augmentation
    - Spectral view: Center pixel spectral signature [6:]
    """

    def __init__(self, data, label, n_bands, transform):
        self.data = data.reshape(-1, 28, 28, n_bands)
        self.label = label
        self.transform = transform
        self.n_classes = label.max() + 1

    def __getitem__(self, i):
        spatial_view1 = Image.fromarray(self.data[i, :, :, :3])
        spatial_view1 = self.transform(spatial_view1)

        spatial_view2 = Image.fromarray(self.data[i, :, :, 3:6])
        spatial_view2 = self.transform(spatial_view2)

        spectral_view = self.data[i, 15, 15, 6:]
        spectral_min, spectral_max = spectral_view.min(), spectral_view.max()
        spectral_view = torch.tensor(
            (spectral_view - spectral_min) / (spectral_max - spectral_min)
        ).to(dtype=spatial_view1.dtype)

        return spatial_view1, spatial_view2, spectral_view, self.label[i]

    def __len__(self):
        return len(self.data)


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(28),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
])
