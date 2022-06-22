import torch
from torchvision import datasets, transforms

def loadDataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307),(0.3081))
    ])

    train_set = datasets.MNIST('DATA_MNIST/', download=True, train=True, transform=transform)
    trainLoader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    validation_set = datasets.MNIST('DATA_MNIST/', download=True, train=False, transform=transform)
    validationLoader = torch.utils.data.DataLoader(validation_set, batch_size=64, shuffle=True)

    return trainLoader, validationLoader, train_set, validation_set