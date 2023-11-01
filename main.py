import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils import load_yaml_file, classification_accuracy, transform
from data_generator import TrainDataGenerator
from model import ConvNetModel
from torch.nn import CrossEntropyLoss

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    settings = load_yaml_file(config.exp + '.yaml')

    # create folder to save models and loss graphs
    reference = settings['train']['net_type'] + str(time.strftime("_%Y%m%d_%H%M%S"))
    checkpoints_folder = settings['data']["output_dir"] + '/checkpoints/' + reference
    os.makedirs(checkpoints_folder, exist_ok=True)

    train_dataset = TrainDataGenerator(settings, 'train', transform('train'))
    val_dataset = TrainDataGenerator(settings, 'val',  transform('val'))

    train_iterator = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=True)
    val_iterator = DataLoader(val_dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=True)

    epochs = settings['train']['num_epochs']

    model = ConvNetModel().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=float(settings['train']['learning_rate']))
    loss_fn = CrossEntropyLoss().to(device)

    classification_loss = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)
    validation_loss = np.zeros(epochs)

    best_accuracy = 0.0

    for epoch in range(epochs):
        c_loss = 0
        acc = 0

        model.train()
        for i, (au_in, target) in enumerate(train_iterator):
            au_in = au_in.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)

            optimizer.zero_grad()
            predictions = model(au_in)

            loss = loss_fn(input=predictions, target=target.long())
            loss.backward()
            optimizer.step()

            c_loss += loss.item()
            acc += classification_accuracy(predictions, target)

        # average loss per epoch
        classification_loss[epoch] = c_loss / (i + 1)
        # average accuracy per epoch
        train_accuracy[epoch] = acc / (i + 1)

        print("epoch = {}, average classification loss ={}".format(epoch, classification_loss[epoch]))
        print("epoch = {}, Training accuracy ={}".format(epoch, train_accuracy[epoch]))

        model.eval()

        with torch.no_grad():
            val_acc = 0
            val_loss = 0
            for i, (au_in, target) in enumerate(val_iterator):
                au_in = au_in.to(device, dtype=torch.float)
                target = target.to(device, dtype=torch.float)

                preds = model(au_in)

                loss = loss_fn(input=preds, target=target.long())
                val_loss += loss.item()
                val_acc += classification_accuracy(preds, target)

        # average loss per epoch
        validation_loss[epoch] = val_loss / (i + 1)
        print("epoch = {}, average validation loss ={}".format(epoch, validation_loss[epoch]))
        val_accuracy[epoch] = val_acc / (i + 1)
        print("epoch = {},  Validation set accuracy ={}".format(epoch, val_accuracy[epoch]))
        print('***********************************************************')

        # plot accuracy curves and save model
        plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'b-', label=" Train Accuracy")
        plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r-', label="Validation Accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend(loc='best')
        plt.savefig(checkpoints_folder + "/accuracy.jpeg", bbox_inches="tight")
        plt.clf()

        # plot loss curves
        plt.plot(range(1, len(classification_loss) + 1), classification_loss, 'b-', label="Classification loss")
        plt.xlabel("epochs")
        plt.ylabel("Error")
        plt.legend(loc='best')
        plt.savefig(checkpoints_folder + "/classification_loss.jpeg", bbox_inches="tight")
        plt.clf()

        plt.plot(range(1, len(validation_loss) + 1), validation_loss, 'b-', label="Validation loss")
        plt.xlabel("epochs")
        plt.ylabel("Error")
        plt.legend(loc='best')
        plt.savefig(checkpoints_folder + "/validation_loss.jpeg", bbox_inches="tight")
        plt.clf()

        # Save the model if it has the best accuracy so far
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

        # if (epoch + 1) % 10 == 0:
        #     net_save = {'net': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}
        #     torch.save(net_save, checkpoints_folder + "/convnet_cifar_epoch{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='config', help='Experience settings YAML file')
    config = parser.parse_args()

    main(config)
