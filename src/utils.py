
import random
import matplotlib.pyplot as plt
import random
import torch
import sys
import os
import config
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.squat import model

def get_random_image(target_class, dataset, seed=None):
    if seed is not None:
        random.seed(seed)

    indices = [i for i, (_, label) in enumerate(dataset) if label == target_class]
    random_index = random.choice(indices)
    random_image, random_label = dataset[random_index]
    random_image = random_image.float()
    return random_image

def show_image(image, title=""):
    plt.figure(figsize=(2, 2))
    plt.title(title)
    plt.imshow(image.squeeze().numpy(), cmap='gray') 
    plt.axis('off')
    plt.show()
  
def plot_tensor(t, title=""):
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

def f(x, x_k):
  x = x.flatten()
  x_k = x_k.flatten()
  return (1/2)*torch.norm(x - x_k, p='fro')**2    

def f_gradient(x, x_k):
  return (x_k - x).flatten().cpu().numpy()

      
''' 
g(xk) = (I_k - 1_K^T*e_j )*C(x_k)
'''
def g(x_k):
  I = torch.eye(config.classes) # identity matrix of size K
  ones_vec = torch.ones(config.classes) # all ones vector of size K
  canonical_vec = torch.zeros(config.classes) # canonical vector of size K
  canonical_vec[config.j] = 1
  g_val = torch.mv((I-(canonical_vec*ones_vec.t())), (model(x_k.view(1, 1, 28, 28))).cpu().numpy())
  return g_val

def lagrangian(x, x_k, λ_k):
  x = x.flatten()
  x_k = x_k.flatten()
  # λ_k = torch.tensor(λ_k, dtype=x.dtype) 
  λ_k = torch.tensor(λ_k, dtype=x.dtype).clone().detach()
  L = ((1/2)*torch.norm(x - x_k, p='fro')**2) + λ_k.t() @ g(x_k)
  return L
