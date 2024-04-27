import torch
from torchvision import datasets as dset
from torchvision import transforms
import matplotlib.pyplot as plt
def change_directory(path):
    import os
    os.chdir(path)

def load_dataset():
    dataset = dset.MNIST(
        root='/data',
        download=True,
        transform=transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,)),#-1,1
        ])
    )
    return dataset
