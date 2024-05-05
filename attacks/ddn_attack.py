from adv_lib.attacks import ddn
import torch
import os
import sys
#import for sleep
from time import sleep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cvxpy as cp
from src import utility_functions
import torchvision
from models.small_cnn import SmallCNN
from tqdm import tqdm
import config
from torchvision import datasets, transforms

if torch.cuda.is_available():
    print("GPU is available")
    device = "cuda"
else:
    print("GPU is not available.")
    device = "cpu"

model = SmallCNN()
model.load_state_dict(torch.load(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/checkpoints/smallcnn_regular/model-nn-epoch10.pt", map_location=torch.device(device)))

transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
adv_samples_10 = ddn(model=model, inputs=data, labels=target, steps=1)
adv_samples_1000 = ddn(model=model, inputs=data, labels=target, steps=1000)

print(f"target: {target}")
utility_functions.show_image(adv_samples_10[1])
utility_functions.show_image(adv_samples_1000[1])


print(f"Prediction: {torch.argmax(model(adv_samples_1000[1]))}")