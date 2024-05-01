
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
import config
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.squat import model

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

# def plot_tensor(t, title = "", dim = 1.0):
#   plt.figure(figsize=(dim, dim))
#   plt.imshow(t.squeeze().numpy(), cmap='gray')  # Use grayscale colormap
#   plt.axis('off')
#   plt.title(title)
#   plt.show()
#   plt.close()
  
def plot_tensor(t, title="", dim=1.0):
    # Converti il tensore in un array NumPy
    img_array = t.squeeze().numpy()

    # Normalizza l'array per convertirlo in un'immagine in scala di grigi (0-255)
    img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
    img_array = cv2.resize(img_array, (300, 300))
    
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.moveWindow(title, 500, 300)
    
    # Mostra l'immagine utilizzando OpenCV
    cv2.imshow(title, img_array)
    cv2.waitKey(200)
    cv2.destroyWindow(title)
    
def g(xk):
  I = torch.eye(config.classes) # identity matrix of size K
  ones_vec = torch.ones(config.classes) # all ones vector of size K
  canonical_vec = torch.zeros(config.classes) # canonical vector of size K
  canonical_vec[config.j] = 1
  g_val = (I-(canonical_vec*ones_vec.t())) * torch.argmax(model(xk.reshape(1,28,28)))
  return g_val

def f(x, xk):
  return (1/2)*torch.norm(x - xk, p='fro')**2

def algebric_f_gradient(x,xk):
  f_gradient = torch.zeros(28*28)
  for i in range(28*28):
    f_gradient[i] = -torch.norm(torch.flatten(x)[i]-torch.flatten(xk)[i]) # gradient of f(x) (vector in R^K)
  return f_gradient

def compute_f_gradient(x, xk):
  if config.ALGEBRIC_GRADIENT:
      f_gradient = algebric_f_gradient(x,xk)
      # utility_functions.plot_tensor(f_gradient.detach().reshape(1,28,28), title=f"f_gradient (k={k})", dim=1.0)
  else:
      xk = torch.tensor(xk.data, requires_grad=True)
      f = (1/2)*torch.norm(x - xk, p='fro')**2 # frobenius_norm between x and xk
      f.backward()
      f_gradient = xk.grad.data
      f_gradient = f_gradient.flatten()
      # utility_functions.plot_tensor(f_gradient.detach().reshape(1,28,28), title=f"f_gradient (k={k})", dim=1.0)
  return f_gradient


def compute_lagrangian(x, xk, lambda_k):
    f_val = f(x, xk)  # Assuming f function exists that computes the objective function value at xk
    g_val = g(xk)
    lagrangian = f_val + torch.dot(lambda_k, g_val)
    return lagrangian


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

