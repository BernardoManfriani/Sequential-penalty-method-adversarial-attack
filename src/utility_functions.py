import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import random
import torch.nn.functional as F
import torch
import cvxpy as cp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.squat import j, model,K

def get_random_image(target_class, dataset):
  random_index = random.choice([i for i, (_, label) in enumerate(dataset) if label == target_class])
  random_image, random_label = dataset[random_index]
  random_image = random_image.float()
  return random_image

def show_image(image):
    plt.figure(figsize=(2, 2))
    plt.title("Original example")
    plt.imshow(image.squeeze().numpy(), cmap='gray')  # Use grayscale colormap
    plt.axis('off')
    plt.show()



def plot_tensor(t, title = "", dim = 1.0):
  plt.figure(figsize=(dim, dim))
  plt.imshow(t.squeeze().numpy(), cmap='gray')  # Use grayscale colormap
  plt.axis('off')
  plt.title(title)
  plt.show()

# def get_random_image_cvxpy(target_class):
#   # Seleziona un'immagine casuale dalla classe specificata
#   random_index = random.choice([i for i, (_, label) in enumerate(dataset) if label == target_class])
#   random_image, _ = dataset[random_index]
#   random_image = random_image.float()  # Converti in float per operazioni future

#   # Visualizzazione dell'immagine originale
#   plt.title("Original example")
#   plt.imshow(random_image.squeeze().numpy(), cmap='gray')
#   plt.axis('off')
#   plt.show()

#   numpy_image = random_image.numpy().flatten()  # Assicurati che sia un vettore 1D
#   cvxpy_variable = cp.Variable(numpy_image.shape, value=numpy_image)
#   return cvxpy_variable  # Restituisce la variabile CVXPY

def g(xk):
  I = torch.eye(K) # identity matrix of size K
  ones_vec = torch.ones(K) # all ones vector of size K
  canonical_vec = torch.zeros(K) # canonical vector of size K
  canonical_vec[j] = 1
  g_val = (I-(canonical_vec*ones_vec.t())) * torch.argmax(model(xk))
  return g_val

def f(x, xk):
  return (1/2)*torch.norm(x - xk, p='fro')**2