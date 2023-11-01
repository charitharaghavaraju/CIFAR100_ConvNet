import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.transforms import transforms


def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='bytes')
    return myDict


def load_data(root_dir, split):
    data = unpickle(root_dir + '/' + split)

    images = data[b'data']
    labels = data[b'fine_labels']
    course_labels = data[b'coarse_labels']

    return images, labels


def load_meta_data(settings):
    metaData = unpickle(settings['data']['root_dir'] + '/meta')

    fine_label_names = metaData['fine_label_names']
    coarse_label_names = metaData['coarse_label_names']

    return fine_label_names, coarse_label_names


def load_yaml_file(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def classification_accuracy(logits, ground_truth):
    n_images = logits.shape[0]
    prediction = torch.argmax(logits, dim=1)
    x = prediction - ground_truth
    n_wrong_predictions = np.count_nonzero(x.cpu().numpy())
    accuracy = (n_images - n_wrong_predictions) / n_images

    return accuracy


def transform(split):
    # Define the transforms to be applied to the data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if split == 'train' or split == 'val':
        return transform_train
    else:
        return transform_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR-100 Classification')
    parser.add_argument('--exp', type=str, default='config', help='Experience settings YAML file')

    config = parser.parse_args()

    file_path = Path(config.exp + '.yaml')
    settings = load_yaml_file(file_path)
    load_data(settings, 'train')
