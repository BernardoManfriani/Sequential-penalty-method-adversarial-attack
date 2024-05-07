import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from time import sleep
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
model.load_state_dict(torch.load(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/checkpoints/smallcnn_regular/model-nn-epoch10.pt", map_location=torch.device(device)))

d = cp.Variable(28*28) # direction (variable of the optimization problem)
I = torch.eye(28*28) # identity matrix
d_lambda = cp.Variable(28*28)
lambda_k = torch.zeros(10)  # Initialize Lagrange multipliers

def squat_algorithm(x, xk):
    utility_functions.show_image(xk, title="x0")

    for k in range (0, config.N_iter):
        
        f_gradient = utility_functions.compute_f_gradient(x, xk)
        
        # FIRST ORDER
        if k<=config.N_1:
            
            # f_gradient = utility_functions.compute_f_gradient(x, xk)
            objective = cp.Minimize(f_gradient.t()@d + 0.5 * cp.quad_form(d, I))

            # CONSTRAINTS - finire
            g_val = cp.Variable(utility_functions.g(xk).shape, value=utility_functions.g(xk).detach().numpy())
            jacobian = torch.autograd.functional.jacobian(utility_functions.g, xk.flatten(), create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            #mostra jacobian 
            print(f"jacobian: {jacobian.shape}")
            
            constraints = [cp.matmul(jacobian,d) + g_val <= 0]
            # constraints = [jacobian[0].T@d + g_val <= 0,
            #     jacobian[1].T@d + g_val <= 0,
            #     jacobian[2].T@d + g_val <= 0,
            #     jacobian[3].T@d + g_val <= 0,
            #     jacobian[4].T@d + g_val <= 0,
            #     jacobian[5].T@d + g_val <= 0,
            #     jacobian[6].T@d + g_val <= 0,
            #     jacobian[7].T@d + g_val <= 0,
            #     jacobian[8].T@d + g_val <= 0,
            #     jacobian[9].T@d + g_val <= 0,
            #     ]
         
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
                # xk = torch.clamp(torch.tensor(xk), 0.0, 1.0)
                xk = torch.tensor(xk)
                xk = xk.reshape(28,28)
                xk = xk.to(torch.float32)
                
            utility_functions.show_image(xk, title=f"x{k+1}")
            # utility_functions.plot_tensor(xk, f"x{k}", dim=2)
            
            # sleep(30)
            # print(f"Model prediction for x_{k+1}: {model(xk.reshape(1,28,28))}")
            print(f"Model prediction for x_{k+1}: {torch.argmax(model(xk.reshape(1,28,28)))}")
            
            if torch.argmax(model(xk.reshape(1,28,28))) == config.j:
                print("SmallCNN has been corrupted")
                utility_functions.show_image(xk)
                k = config.N_1+1
        
        # SECOND ORDER
        
        if k>config.N_1:
           
            lagrangian = utility_functions.compute_lagrangian(x, xk, lambda_k)
            print(f"lagrangian.shape: {lagrangian.shape}")
            hessian_lagrangian = torch.autograd.functional.hessian(utility_functions.compute_lagrangian, (x.flatten(), xk.flatten(), lambda_k), create_graph=False)
            
            # Ensure the Hessian is a numpy array to use in CVXPY
            # hessian_lagrangian_np = hessian_lagrangian[0][0].detach().numpy()
            
            objective = cp.Minimize(f_gradient.T @ d + 0.5 * cp.quad_form(d_lambda, hessian_lagrangian))
            g_val = utility_functions.g(xk).detach().numpy()
            jacobian = torch.autograd.functional.jacobian(utility_functions.g, xk).detach().numpy()
            
            # Define constraints (assuming jacobian is correctly computed)
            # constraints = [jacobian.T @ d + g_val <= 0]
            constraints = [cp.matmul(jacobian,d) + g_val <= 0]
            
            # Solve the quadratic problem
            problem = cp.Problem(objective, constraints)
            result = problem.solve()
            d_x = d.value
            d_λ = d_lambda.value
            
            # Update:  xk+1 ← xk + βdx
            xk = xk.detach().numpy().flatten() + config.beta * d_x
            # xk = torch.from_numpy(xk.detach().numpy().flatten() + optimal_d).view_as(xk)
            
            # Update: λ_k+1 ← β*d_λ
            lambda_k = config.beta * d_λ

            # utility_functions.plot_tensor(xk, f"x_{k+1}", dim=2)
            utility_functions.show_image(xk)
            print(f"Model prediction for x_{k+1}: {torch.argmax(model(xk.reshape(1,28,28)))}")
                
    utility_functions.show_image(xk, title=f"x_{k}")

    
