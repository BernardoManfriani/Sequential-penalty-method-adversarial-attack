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

'''
    From the constrained problem
        min_d 0.5 ||x-xk||^2
        g(xk) ≤ 0
        
    To the unconstrained subproblem
        min_ 0.5 ||x-xk||^2 + τ · max{0, g(xk)}^2
        min_ 0.5 ||x-xk||^2 + τ · sum_i^m (max{0, g(xk)}^2)
        
'''
x = utils.get_random_image(config.target_class, seed=111)
xk = utils.get_random_image(config.original_class, seed=111)

# xk = cp.Variable(28*28)

f = 0.5 * np.linalg.norm(x-xk)**2
tau = 1
g = utils.g(xk)
F = f + tau * np.sum(np.max(0, g))
print(F)

# objective = cp.Minimize(F)
# problem = cp.Problem(objective)
# problem.solve

# utils.show_image(xk.value)

