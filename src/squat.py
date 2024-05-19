import torch
import os
import sys
import numpy as np
import cvxpy as cp
from cvxpy import OSQP

# Add project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src import utils
from models.small_cnn import SmallCNN
import config

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU is {'available' if device == 'cuda' else 'not available'}.")

# Load the model
model = SmallCNN()
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'smallcnn_regular', 'model-nn-epoch10.pt'),
    map_location=torch.device(device)
))

def squat_algorithm(x, x_k):
    n = 28 * 28
    d = cp.Variable(n)  # direction (variable of the optimization problem)
    λ_k = np.zeros(10)  # initialize Lagrange multipliers
    utils.show_image(x_k, title="x0")
    I = torch.eye(n)
    
    for k in range(config.N_iter):
        # FIRST ORDER
        if k < config.N_1:
            f_grad = utils.f_gradient(x, x_k)
            objective = cp.Minimize(cp.matmul(f_grad.t(), d) + 0.5 * cp.quad_form(d, I))
            
            # CONSTRAINTS
            g = cp.Variable(utils.g(x_k).shape, value=utils.g(x_k).numpy())
            g_grad = torch.autograd.functional.jacobian(utils.g, x_k.flatten()).numpy()
            constraints = [g_grad @ d <= -g]  # Ensuring constraints are correctly defined
            
            # QP SUB-PROBLEM
            problem = cp.Problem(objective, constraints)
            result = problem.solve(solver=cp.OSQP, verbose=False)
            optimal_d = d.value
            
            # UPDATE
            x_k = x_k.flatten() + (config.α * optimal_d)
            x_k = x_k.clamp(min=0.0, max=1.0).reshape(28, 28).to(torch.float32)
            
            utils.show_image(x_k, title=f"x{k+1}")
            print(f"||x - x_k||: {torch.norm(x.flatten() - x_k.flatten(), p='fro')}")
            
            logits = model(x_k.unsqueeze(0))
            print(f"logits: {logits.detach().numpy()}")
            print(f"Model prediction for x_{k+1}: {torch.argmax(logits)}")
            
            if torch.argmax(logits) == config.j:
                print("SmallCNN has been corrupted")
        
        # SECOND ORDER
        else:
            f_grad = utils.f_gradient(x, x_k)
            
            ''' PRIMO MODO '''
            L_hessian = torch.autograd.functional.hessian(utils.lagrangian, inputs=(x, x_k, torch.from_numpy(λ_k)))[0][0]
            L_hessian = L_hessian.view(784, 784)
            
            ''' SECONDO MODO '''
            # def wrapped_function(x_k):
            #     return utils.lagrangian(x, x_k, λ_k)
            # L_hessian = torch.autograd.functional.hessian(wrapped_function, x_k.flatten()) # (784, 784)
            
            objective = cp.Minimize(f_grad.numpy().T @ d + 0.5 * cp.quad_form(d, L_hessian))
            
            # CONSTRAINTS
            g = cp.Variable(utils.g(x_k).shape, value=utils.g(x_k).numpy())
            g_grad = torch.autograd.functional.jacobian(utils.g, x_k.flatten()).numpy()
            constraints = [g_grad @ d + g <= 0]
            
            # QP SUB-PROBLEM
            problem = cp.Problem(objective, constraints)
            result = problem.solve()
            d_x = d.value # problem solution
            d_λ = constraints[0].dual_value ; print(f"d_λ: {d_λ}") # lagrangian multipliers
            
            # Update x_k and λ_k
            x_k = x_k.flatten() + config.β * d_x
            x_k = x_k.clamp(min=0.0, max=1.0).reshape(28, 28).to(torch.float32)
            λ_k = config.β * d_λ ; print(f"λ_k: {λ_k}")
            
            utils.show_image(x_k, title=f"Second order x{k+1}")
            print(f"||x - x_k||: {torch.norm(x.flatten() - x_k.flatten(), p='fro')}")
            
            logits = model(x_k.unsqueeze(0))
            print(f"logits: {logits.detach().numpy()}")
            print(f"Model prediction for x_{k+1}: {torch.argmax(logits)}")
    
    utils.show_image(x_k, title=f"x_{k}")

