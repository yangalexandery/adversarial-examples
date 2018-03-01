import os
# import h5py
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
# from torchvision import transforms
import transforms
# import affine_transforms
import torch
import time


class CIFAR(Dataset):
    def __init__(self, data_path, split, augment=True, load_everything=True):
        self.count = 0
        file_path = os.path.join(data_path, 'cifar-10.npz')
        full_data = np.load(file_path)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        transform = [
            transforms.Scale(256),
            transforms.RandomCrop(224),
            # transforms.RandomResizedCrop(224)
            ]

        self.split = split
        if split == 'train':
            self.dataset = full_data['arr_0'][:50000]
            self.dataset = self.dataset.reshape((50000, 3, 32, 32))
            self.labels = full_data['arr_1'][:50000]
        else:
            self.dataset = full_data['arr_0'][50000:]
            self.dataset = self.dataset.reshape((10000, 3, 32, 32))
            self.labels = full_data['arr_1'][50000:]
        self.dataset = np.transpose(self.dataset, (0, 2, 3, 1))
        print(self.dataset.shape)

        # if augment:
        #     transform.extend([
        #     transforms.ColorJitter(brightness=0.1, contrast=0.0, saturation=0.3, hue=0.05),
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.RandomVerticalFlip(),
        #     ])

        transform += [transforms.ToTensor()]

        # if augment:
        #     transform.append(
        #         affine_transforms.Affine(rotation_range=5.0, zoom_range=(0.85, 1.0), fill_mode='constant')
        #     )
        # if augment:
        #     transform.append(
        #     affine_transforms.Affine(rotation_range=10.0, translation_range=0.1, zoom_range=(0.5, 1.0), fill_mode='constant')
        #     )

        transform += [
            self.normalize]

        self.preprocess = transforms.Compose(transform)

        # self.split = split
        # if split != 'test':
        #     self.labels = np.array(self.dataset['labels'])
        # self.load_everything = load_everything
        # if self.load_everything:
        #     self.images = np.array(self.dataset['images'])

    def __getitem__(self, index):
        self.count += 1

        # if self.load_everything:
        #     image = self.images[index]
        # else:
        image = self.dataset[index]
        img_tensor = self.preprocess(Image.fromarray(image))

        # if self.split == 'test':
        #     return img_tensor, index
        # img_tensor = self.dataset[index]

        label = self.labels[index]
        label_tensor = torch.LongTensor(np.array([label]).astype(int))

        return img_tensor, label_tensor

    def __len__(self):
        return self.dataset.shape[0]