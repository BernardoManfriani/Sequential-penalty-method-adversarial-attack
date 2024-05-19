from adv_lib.attacks import alma
import torch
import os
import sys
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import shutil
from src import utility_functions
import torchvision
from models.small_cnn import SmallCNN
from torchvision import transforms

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
        
mnist_X_adv = alma(model=model, inputs=data, labels=target, num_steps=4000)
print(mnist_X_adv.type)
print(f"target: {target}")
utility_functions.show_image(mnist_X_adv[0])
print(f"Prediction: {torch.argmax(model(mnist_X_adv[0]))}")

mnist_X_adv_array = mnist_X_adv.numpy()

# np.save('Adversarial-attacks-via-Sequential-Quadratic-Programming/data_attack/alma/mnist_X_adv.npy', mnist_X_adv_array)
np.save(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/data_attack/alma/mnist_X_adv.npy", mnist_X_adv_array)

# image_array = np.load(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/data_attack/alma/mnist_X_adv.npy")

# image_normalized = (image_array * 255).astype(np.uint8)

# # Salva l'immagine come file PNG utilizzando OpenCV
# cv2.imwrite('immagine.png', image_normalized)