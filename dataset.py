import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from skimage import transform
from config import config

class ShapesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_path = os.listdir(root_dir)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.images_path[idx])
        image = Image.open(image_path)
        if self.transform:
            sample = self.transform(image)
        return sample

def get_dataloader():
    train_transformed_dataset = ShapesDataset(root_dir = config.dataset_dir,
                                               transform = transforms.Compose([
                                                   transforms.Resize(config.imageSize),
                                                   transforms.ToTensor()
                                               ]))

    dataloader = DataLoader(train_transformed_dataset, batch_size = config.batch_size, shuffle = True, **config.kwargs)
    return dataloader

def show_batch(sample_batched):
    images_batch = sample_batched.numpy()
    batch_size = len(images_batch)
    for i in range(5):
        plt.figure()
        plt.tight_layout()
        plt.imshow(np.squeeze(images_batch[i].transpose((1, 2, 0))))
        plt.show()

if __name__ == '__main__':
    data_loader = get_dataloader()
    for i, images in enumerate(data_loader):
        print(images.shape)
        show_batch(images)