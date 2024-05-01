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
        
        lambda_k = torch.zeros(utility_functions.g(xk).shape, device=x.device)  # Initialize Lagrange multipliers
        f_gradient = utility_functions.compute_f_gradient(x, xk)

        # FIRST ORDER
        if k<=config.N_1:
            
            # f_gradient = utility_functions.compute_f_gradient(x, xk)
            objective = cp.Minimize(f_gradient.t()@d + 0.5 * cp.quad_form(d, I))

            # CONSTRAINTS - finire
            g_val = cp.Variable(utility_functions.g(xk).shape, value=utility_functions.g(xk).detach().numpy())
            jacobian = torch.autograd.functional.jacobian(utility_functions.g, xk, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            #mostra jacobian 
            # print(f"jacobian: {jacobian.value}")
            constraints = [g_val <= 0]
            # constraints = [jacobian.T@d + g_val <= 0]

            # QP PROBLEM
            problem = cp.Problem(objective, constraints) # definition of the constrained problem to minimize
            result = problem.solve()
            optimal_d = d.value
            # print(f"d.value: {d.value}")

            #somma tutti i valori di d per capire se da sempre 0
            # sum_d = 0
            # for i in range(len(optimal_d)):
            #     sum_d += optimal_d[i]
            # print(f"sum_d: {sum_d}")


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
            # sleep(30)
            # print(f"Model prediction for x_{k+1}: {model(xk.reshape(1,28,28))}")
            print(f"Model prediction for x_{k+1}: {torch.argmax(model(xk.reshape(1,28,28)))}")

        if k>config.N_1:
            # # SECOND ORDER
            # lagrangian =    
            # hessian_lagrangian = torch.autograd.functional.hessian(lagrangian, xk, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            # objective = cp.Minimize(f_gradient.t()@d + 0.5 * cp.quad_form(d, hessian_lagrangian))

            lagrangian = utility_functions.compute_lagrangian(x, xk, lambda_k)
            hessian_lagrangian = torch.autograd.functional.hessian(utility_functions.compute_lagrangian, (xk, lambda_k), create_graph=False)
            
            # Ensure the Hessian is a numpy array to use in CVXPY
            hessian_lagrangian_np = hessian_lagrangian[0][0].detach().numpy()
            
            # f_gradient = utility_functions.compute_f_gradient(x, xk).detach().numpy()
            # d = cp.Variable(xk.numel())  # direction (variable of the optimization problem)
            
            objective = cp.Minimize(f_gradient.T @ d + 0.5 * cp.quad_form(d, hessian_lagrangian_np))
            g_val = utility_functions.g(xk).detach().numpy()
            jacobian = torch.autograd.functional.jacobian(utility_functions.g, xk).detach().numpy()
            
            # Define constraints (assuming jacobian is correctly computed)
            constraints = [jacobian.T @ d + g_val <= 0]
            
            # Solve the quadratic problem
            problem = cp.Problem(objective, constraints)
            result = problem.solve()
            optimal_d = d.value
            
            # Update xk based on the direction found
            xk = torch.from_numpy(xk.detach().numpy().flatten() + optimal_d).view_as(xk)
            
            # Update lambda_k based on some dual update rule (not explicitly defined here)
            lambda_k += config.alpha_dual * torch.from_numpy(g_val)  # Update rule for lambda_k needs to be defined appropriately

            utility_functions.plot_tensor(xk, f"x_{k+1}", dim=2)
            print(f"Model prediction for x_{k+1}: {torch.argmax(model(xk.reshape(1,28,28)))}")
                
            # SECOND ORDER
            # torch.autograd.functional.hessian or torch.autograd.functional.jacobian