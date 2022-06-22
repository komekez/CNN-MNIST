import torch
import numpy as np
from matplotlib import pyplot as plt

from torchvision import datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F

from datasetHelper import loadDataset
import neural


def main():
    train_loader, validation_loader, train_set, val_set = loadDataset()

    training_data = enumerate(train_loader)
    batch_idx, (images, labels) = next(training_data)

    print(type(images)) # Checking the datatype 
    print(images.shape) # the size of the image
    print(labels.shape) # the size of the labels

    model = neural.neuralNetwork()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()


    epochs = 20
    train_loss, val_loss = [], []
    accuracy_total_train, accuracy_total_val = [], []

    for epoch in range(epochs):
        total_train_loss =0 
        total_val_loss = 0

        model.train()

        total = 0
        for idx, (image, label) in enumerate(train_loader):
            image,label = image, label
            optimizer.zero_grad()

            pred = model(image)
            loss = criterion(pred, label)

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            pred = torch.nn.functional.softmax(pred, dim=1)

            for i, p in enumerate(pred):
                if(label[i] == torch.max(p.data, 0)[1]):
                    total += 1

        accuracy_train = total/(len(train_set))
        accuracy_total_train.append(accuracy_train)

        total_val_loss = total_val_loss/(idx+1)
        val_loss.append(accuracy_train)

        if epoch % 5 == 0:
            print("Epoch: {}/{}  ".format(epoch, epochs),
                "Training loss: {:.4f}  ".format(total_train_loss),
                "Testing loss: {:.4f}  ".format(total_val_loss),
                "Train accuracy: {:.4f}  ".format(accuracy_train),
                "Test accuracy: {:.4f}  ".format(accuracy_val))



main()


