import torch
import numpy as np
from matplotlib import pyplot as plt

from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F

from datasetHelper import loadDataset


def main():
    train_data, validation_data = loadDataset()

main()


