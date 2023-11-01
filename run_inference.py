import argparse

import torch
from model import ConvNetModel
import utils
import pickle
from torch.utils.data import DataLoader
from data_generator import TestDataGenerator
import numpy as np
import os
import csv


def run_inference(config, model_path):
    model = ConvNetModel().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)

    settings = utils.load_yaml_file(config.exp + '.yaml')

    batch_size = 100
    test_dataset = TestDataGenerator(settings, 'test', utils.transform('test'))
    test_iterator = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=settings['train']['num_workers'],
                          pin_memory=True,
                          shuffle=False, drop_last=True)

    model.eval()
    test_accuracy = 0.0

    with torch.no_grad():
        for i, (au_in, target) in enumerate(test_iterator):
            au_in = au_in.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            preds = model(au_in)
            _, predicted = torch.max(preds.data, 1)
            test_accuracy += utils.classification_accuracy(preds, target)

    # accuracy = 100 * test_accuracy
    print('Accuracy on test set: %.2f %%' % test_accuracy)


if __name__ == '__main__':
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_dir = " "  # example: "checkpoints/ResNet_20201108_222637/"
    model_path = "best_model.pth"  # example: "pretrained_model.pth"

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='config', help='Experience settings YAML file')
    config = parser.parse_args()

    run_inference(config, model_path)
