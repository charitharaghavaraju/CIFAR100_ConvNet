# CIFAR100 Classification using Simple ConvNet

This repository is a simple Pytorch Implementation of Convolutional Neural Networks 
to implement classification on CIFAR100 dataset.

Requirements: Pytorch, PIL, Matplotlib, Pickle

## Dataset

Download and extract the dataset from the [official website of CIFAR100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)

## Training

1. Update the dataset path `root_dir` in `config.yaml`
2. Run `main.py`
3. Save the best model.

## Inference

1. Update the saved best model in the `run_inference.py`
2. Run `run_inference.py`