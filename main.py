import torch
import numpy as np
from matplotlib import pyplot as plt

from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F

from datasetHelper import loadDataset


def main():
    train_loader, validation_loader = loadDataset()

    training_data = enumerate(train_loader)
    batch_idx, (images, labels) = next(training_data)

    print(type(images)) # Checking the datatype 
    print(images.shape) # the size of the image
    print(labels.shape) # the size of the labels

    

main()


