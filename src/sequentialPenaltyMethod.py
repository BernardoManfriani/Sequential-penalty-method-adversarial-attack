import torch
import os
import sys
import numpy as np
# import cvxpy as cp
from src import data_preparation
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Add project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import utils
from models.small_cnn import SmallCNN
import config

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallCNN().to(device)
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'smallcnn_regular', 'model-nn-epoch10.pt'),
    map_location=device
))

dataset = data_preparation.load_dataset()

#C: Rn -> Rk classificatore che assegna a un input x la label i (la classe i è scelta poiché Ci(x) è il maggiore)
def C(x):
    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float32).to(device)
        x_reshaped = x.view(1, 1, 28, 28)
        return model(x_reshaped).cpu().numpy()

# Define the objective function
def objective(x, x_orig):
    return 0.5 * np.linalg.norm(x - x_orig)**2

# Define the penalty function P(x) = max(0,g(x))^2 * τ
def penalty(x, x_orig, target_class, tau):
    logits = C(x)
    target_logit = logits[0, target_class]
    max_non_target_logit = np.max(logits[0, np.arange(logits.shape[1]) != target_class])
    return max(0, max_non_target_logit - target_logit)**2 * tau

# Define the unconstrained objective
def unconstrained_objective(x, x_orig, target_class, tau):
    return objective(x, x_orig) + penalty(x, x_orig, target_class, tau)

# Generate adversarial example
def generate_adversarial_example(x_orig, target_class, tau_initial=200000000, tau_increase_factor=10, max_iter=2):
    #stampa classificazione x_orig`
    logits = C(x_orig)
    print(f"Classificazione iniziale: {np.argmax(logits)}")
    x_adv = x_orig.clone().detach().numpy()
    x_orig_np = x_orig.clone().detach().numpy()
    tau = tau_initial
    #print (x_orig_np)
    print("inizio")
    for i in range(max_iter):
        utils.plot_tensor(torch.tensor(x_adv), title="x_adv")
        #printa questa iterazione
        print(f"Iterazione {i+1}/{max_iter}")
        res = minimize(unconstrained_objective, x_adv.flatten(), args=(x_orig_np.flatten(), target_class, tau), method='L-BFGS-B')
        x_adv = res.x.reshape(28, 28)
        logits = C(x_adv)
        print(f"Logits: {logits}")
        print(f"Classificazione: {np.argmax(logits)}")
        if np.argmax(logits) == target_class:
            break
        tau *= tau_increase_factor

    return torch.tensor(x_adv, dtype=torch.float32)