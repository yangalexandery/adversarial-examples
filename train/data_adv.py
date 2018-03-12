import os
# import h5py
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
# from torchvision import transforms
import transforms
# import affine_transforms
import time
from model_simple import SimpleConvNet



class CIFAR(Dataset):
    def __init__(self, data_path, split, augment=True, load_everything=True, filename='cifar-10.npz'):
        self.count = 0
        file_path = os.path.join(data_path, filename)
        full_data = np.load(file_path)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        transform = [
            # transforms.Scale(256),
            # transforms.RandomCrop(224),
            # transforms.RandomResizedCrop(224)
            ]

        self.split = split
        self.dataset = full_data['arr_0']
        self.labels = full_data['arr_1']
        if split == 'train':
            if filename == 'cifar-10.npz':
                self.dataset = self.dataset[:50000]
                self.labels = self.labels[:50000]
            self.dataset = self.dataset.reshape((50000, 3, 32, 32))
        else:
            if filename == 'cifar-10.npz':
                self.dataset = full_data[50000:]
                self.labels = self.labels[50000:]
            self.dataset = self.dataset.reshape((10000, 3, 32, 32))
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
        self.to_image = transforms.ToPILImage()



    def __getitem__(self, index):
        self.count += 1

        # img_tensor = torch.FloatTensor(self.dataset[index])
        # img_tensor = Image.fromarray(self.dataset[index])
        image = self.dataset[index].astype(np.uint8)
        # print(image.shape).astype(uint8)
        img_tensor = self.preprocess(Image.fromarray(image))
        # img_tensor = self.preprocess(self.to_image(img_tensor))
        # print(img_tensor.squeeze(0).shape)
        # print(adv_output.size())
        # if self.split == 'test':
        #     return img_tensor, index
        # img_tensor = self.dataset[index]
        label = self.labels[index]
        label_tensor = torch.LongTensor(np.array([label]).astype(int))
        # print(adv_input.grad.data)

        return img_tensor, label_tensor

    def __len__(self):
        return self.dataset.shape[0]