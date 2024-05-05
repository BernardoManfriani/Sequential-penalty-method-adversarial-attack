import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from models.small_cnn import SmallCNN
from attacks import squat_attack_mnist

if torch.cuda.is_available():
    print("GPU is available")
    device = "cuda"
else:
    print("GPU is not available.")
    device = "cpu"
    


# Definizione della funzione per calcolare la robust accuracy utilizzando l'attacco "squat"
def calculate_robust_accuracy(model, testloader, epsilon):
    correct = 0
    total = 0
    for images, labels in testloader:
        images.requires_grad = True
        output = model(images)
        loss = nn.CrossEntropyLoss()(output, labels)
        model.zero_grad()
        loss.backward()

        # Calcolo dell'attacco "squat"
        perturbed_images = squat_attack_mnist(images, labels)

        # Classificazione delle immagini perturbate
        perturbed_output = model(perturbed_images)
        _, perturbed_predicted = torch.max(perturbed_output.data, 1)

        total += labels.size(0)
        correct += (perturbed_predicted == labels).sum().item()

    robust_accuracy = 100 * correct / total
    return robust_accuracy

def plot_robust_accuracy_curve(model, testloader):
    epsilon_values = np.linspace(0, 5, num=50)
    robust_accuracies = []

    for epsilon in epsilon_values:
        robust_accuracy = calculate_robust_accuracy(model, testloader, epsilon)
        robust_accuracies.append(robust_accuracy)

    plt.plot(epsilon_values, robust_accuracies)
    plt.xlabel('L2 Norm')
    plt.ylabel('Robust Accuracy (%)')
    plt.title('Robust Accuracy Curve')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    model = SmallCNN()
    model.load_state_dict(torch.load(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/checkpoints/smallcnn_regular/model-nn-epoch10.pt", map_location=torch.device(device)))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    images = torch.stack([image for image, _ in testset[:500]])  # Rimuovi il `.squeeze()`
    labels = torch.tensor([label for _, label in testset[:500]])
    testloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=1, shuffle=True)
    plot_robust_accuracy_curve(model, testloader)