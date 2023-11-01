import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data.dataset import Dataset
import utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image


class TrainDataGenerator(Dataset):

    def __init__(self, settings, split='train', transform=None):
        super(TrainDataGenerator, self).__init__()

        self.split = split
        self.transform = transform

        self.root_dir = settings['data']['root_dir']
        self.train_images, self.train_labels = utils.load_data(self.root_dir, 'train')

        self.images, self.labels = [], []

        if self.split == 'train':
            self.images = self.train_images[:int(0.8 * len(self.train_images))]
            self.labels = self.train_labels[:int(0.8 * len(self.train_labels))]
        elif self.split == 'val':
            self.images = self.train_images[int(0.8 * len(self.train_images)):]
            self.labels = self.train_labels[int(0.8 * len(self.train_labels)):]

        # self.images = self.images.reshape(len(self.images), 3, 32, 32)
        self.total_images = self.images.shape[0]

    def __len__(self):
        return self.total_images

    def __getitem__(self, item):
        image = self.images[item]
        label = self.labels[item]

        img = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))
        img = Image.fromarray(img)

        if self.transform is not None:
            image = self.transform(img)

        return image, label


class TestDataGenerator(Dataset):

    def __init__(self, settings, split='test', transform=None):
        super(TestDataGenerator, self).__init__()

        self.split = split
        self.transform = transform

        self.root_dir = settings['data']['root_dir']
        self.test_images, self.test_labels = utils.load_data(self.root_dir, 'test')

        self.total_images = self.test_images.shape[0]

    def __len__(self):
        return self.total_images

    def __getitem__(self, item):
        image = self.test_images[item]
        label = self.test_labels[item]

        img = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))

        if self.transform is not None:
            image = self.transform(img)

        return image, label

