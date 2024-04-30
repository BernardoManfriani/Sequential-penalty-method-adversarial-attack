
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cvxpy as cp
from src import utility_functions
import torchvision
from models.small_cnn import SmallCNN
from tqdm import tqdm
import config

if torch.cuda.is_available():
    print("GPU is available")
    device = "cuda"
else:
    print("GPU is not available.")
    device = "cpu"

model = SmallCNN()
# model.load_state_dict(torch.load("training/smallcnn_regular/model-nn-epoch10.pt")) #For CUDA
model.load_state_dict(torch.load(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/training/smallcnn_regular/model-nn-epoch10.pt", map_location=torch.device(device)))

d = cp.Variable(28*28) # direction (variable of the optimization problem)
I = torch.eye(28*28) # identity matrix

def squat_algorithm(x, xk):
    for k in range (0, config.N_iter):
        # FIRST ORDER
        if k<=config.N_1:
            if config.ALGEBRIC_GRADIENT:
                f_gradient = utility_functions.algebric_f_gradient(x,xk)
                # utility_functions.plot_tensor(f_gradient.detach().reshape(1,28,28), title=f"f_gradient (k={k})", dim=1.0)
            else:
                xk = torch.tensor(xk.data, requires_grad=True)
                f = (1/2)*torch.norm(x - xk, p='fro')**2 # frobenius_norm between x and xk
                f.backward()
                f_gradient = xk.grad.data
                f_gradient = f_gradient.flatten()
                # utility_functions.plot_tensor(f_gradient.detach().reshape(1,28,28), title=f"f_gradient (k={k})", dim=1.0)
              
            objective = cp.Minimize(f_gradient.t()@d + 0.5 * cp.quad_form(d, I))

            # CONSTRAINTS - finire
            g_val = cp.Variable(utility_functions.g(xk).shape, value=utility_functions.g(xk).detach().numpy())
            jacobian = torch.autograd.functional.jacobian(utility_functions.g, xk, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            constraints = [g_val <= 0]

            # QP PROBLEM
            problem = cp.Problem(objective, constraints) # definition of the constrained problem to minimize
            result = problem.solve()
            optimal_d = d.value
            # print(f"d.value: {d.value}")

            # UPDATE
            if config.ALGEBRIC_GRADIENT:
                xk = xk.flatten() + config.alpha * optimal_d
                xk = xk.reshape(28,28)
                xk = xk.to(torch.float32)
            else:
                xk = xk.detach().numpy().flatten() + config.alpha * optimal_d
                xk = torch.clamp(torch.tensor(xk), 0.0, 1.0)
                xk = xk.reshape(28,28)
                xk = xk.to(torch.float32)
                
            utility_functions.plot_tensor(xk, f"x{k+1}", dim=2)
            print(f"Model prediction for x_{k+1}: {torch.argmax(model(xk.reshape(1,28,28)))}")

        # SECOND ORDER
        # torch.autograd.functional.hessian or torch.autograd.functional.jacobian


