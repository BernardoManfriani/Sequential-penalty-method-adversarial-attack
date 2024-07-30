import torch
import os
import sys
import numpy as np
import cvxpy as cp
from src import data_preparation
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import data_preparation
from src import utils
from models.small_cnn import SmallCNN
import config

dataset = data_preparation.load_dataset()

# Carica il modello
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN().to(device)
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'smallcnn_regular', 'model-nn-epoch10.pt'),
    map_location=device
))

def C(x):
    with torch.no_grad():
        x = x.to(device)
        x_reshaped = x.view(1, 1, 28, 28)
        return model(x_reshaped).cpu().numpy()

def g(x_k):
    I = torch.eye(10) # matrice (10, 10)
    ones = torch.ones(10) # vettore riga (1,10)
    ej = torch.zeros(10) # vettore riga (1,10)
    ej[config.target_class] = 1
    C_xk = model(x_k.view(1, 1, 28, 28).to(device)) # vettore riga (1,10)
    g = (I - torch.mul(ej.T, ones)) @ C_xk.t().cpu() # g: Rn --> R10
    return g

def objective(d, f_grad, g_val, tau):
    d_torch = torch.tensor(d, dtype=torch.float32, device=device) # converte d in un tensore
    I = torch.eye(d_torch.size(0), device=device)  # matrice identità di dimensione n
    f_grad_torch = torch.tensor(f_grad, dtype=torch.float32, device=device) # converte f_grad in un tensore
    g_val_torch = g_val.to(device) # converte g_val in un tensore
    term1 = torch.dot(f_grad_torch, d_torch) # calcola il prodotto scalare tra f_grad e d
    term2 = 0.5 * torch.dot(d_torch, torch.mv(I, d_torch)) # calcola il prodotto scalare tra d e I*d
    term3 = tau * torch.sum(torch.max(torch.zeros_like(g_val_torch), g_val_torch)**2) # calcola il termine di penalizzazione
    return (term1 + term2 + term3).cpu().item()

def spm(x_origin, target_class, tau=1, tau_increase_factor=10, max_iter=100):
    x_adv = x_origin.clone().detach().to(device)
    x_origin = x_origin.to(device)

    for iteration in range(max_iter):
        logits = C(x_adv)
        pred_class = np.argmax(logits)

        if pred_class == target_class:
            print(f"Success: Adversarial example found after {iteration + 1} iterations.")
            break

        # Calcola la derivata di f con il jacobian
        f_grad = utils.f_gradient(x_origin, x_adv)
        # Calcola g(x_k)
        g_val = g(x_adv)
        # Calcola il gradiente di g con il jacobian
        g_grad = torch.autograd.functional.jacobian(g, x_adv)

        # Stampa x_target, x_origin_clone, f_grad, g_val, g_grad
        print(f"Target class: {target_class}")
        print(f"Original image:")
        # utils.show_image(x_origin.cpu())  # Sposta il tensore sulla CPU per la visualizzazione
        print(f"Adversarial image:")
        # utils.show_image(x_adv.cpu())  # Sposta il tensore sulla CPU per la visualizzazione
        print(f"Gradient of f: {f_grad}")
        print(f"g(x): {g_val}")
        print(f"Gradient of g: {g_grad}")
        
        # Esegui l'ottimizzazione per trovare d ottimale
        optimal_d = np.zeros(784)
        res = minimize(objective, optimal_d, args=(f_grad, g_val, tau), method='L-BFGS-B')
        print(f"Optimization result: {res}")
        optimal_d = torch.tensor(res.x, dtype=torch.float32, device=device).view(1, 28, 28)

        # Aggiorna x_adv
        x_adv = x_adv + optimal_d

        # Aumenta tau
        tau *= tau_increase_factor

    if pred_class != target_class:
        print("Failed to find an adversarial example within the maximum number of iterations.")
    
    return x_adv
