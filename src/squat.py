import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cvxpy as cp
from src import utils
from models.small_cnn import SmallCNN
import config

if torch.cuda.is_available():
    print("GPU is available")
    device = "cuda"
else:
    print("GPU is not available.")
    device = "cpu"

model = SmallCNN()
model.load_state_dict(torch.load(f"{os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}/checkpoints/smallcnn_regular/model-nn-epoch10.pt", map_location=torch.device(device)))

def squat_algorithm(x, x_k):
    
    d = cp.Variable(28*28) # direction (variable of the optimization problem)
    λ_k = torch.zeros(10, requires_grad=True)  # initialize Lagrange multipliers
    utils.show_image(x_k, title="x0")
    I = torch.eye(28*28)
    
    for k in range (0,   config.N_iter):

        # FIRST ORDER
        if k<=config.N_1:
            
            f_grad = utils.f_gradient(x, x_k)
            objective = cp.Minimize(f_grad.numpy().T @ d + 0.5 * cp.norm(d)**2 )  # cp.quad_form(d,I)==cp.norm(d)**2

            # CONSTRAINTS
            ''' PARTE CHE NON MI TORNA'''
            g = cp.Variable(utils.g(x_k).shape, value=utils.g(x_k).numpy())
            g_grad = torch.autograd.functional.jacobian(utils.g, x_k.flatten(), create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            constraints = [g_grad.numpy() @ d + g <= 0]
            
            ''' PROVO A NON DEFINIRE g_val (non funzia)'''
            # g_grad = torch.autograd.functional.jacobian(utils.g, x_k.flatten(), create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            # g = utils.g(x_k).numpy()
            # constraints = [g_grad.numpy().T @ d + g <= 0] # sul paper: g = g_grad.T @ d + g <= 0
            
            # QP SUB-PROBLEM
            problem = cp.Problem(objective, constraints) 
            problem.solve(verbose=True) # a default usa il solver ECOS 
            optimal_d = d.value
            # print(f"d.value: {d.value}")

            # UPDATE
            x_k = x_k.flatten() + (config.α * optimal_d)
            x_k.data = x_k.data.clamp(min=0.0, max=1.0) # porta i valori negativi a 0 e i maggiori di 1 a 1
            x_k = x_k.reshape(28,28)
            x_k = x_k.to(torch.float32)
                
            # utils.show_image(x_k, title=f"x{k+1}")
            utils.plot_tensor(x_k, title=f"x{k+1}")
            
            print(f"Model prediction for x_{k+1}: {torch.argmax(model(x_k.reshape(1,28,28)))}")
            
            if torch.argmax(model(x_k.reshape(1,28,28))) == config.j:
                print("SmallCNN has been corrupted")
        
        # SECOND ORDER
        
        if k>config.N_1:
            
            L = utils.Lagrangian(x, x_k, λ_k)
            L_hessian = torch.autograd.functional.hessian(utils.Lagrangian, (x, x_k, λ_k))
            objective = cp.Minimize(f_grad.numpy().T @ d + 0.5 * cp.quad_form(d, L_hessian))
            
            # CONSTRAINTS
            ''' PARTE CHE NON MI TORNA '''
            g = cp.Variable(utils.g(x_k).shape, value=utils.g(x_k).numpy())
            g_grad = torch.autograd.functional.jacobian(utils.g, x_k.flatten(), create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            constraints = [g_grad.numpy() @ d + g <= 0]
            
            ''' NON FUNZIA '''
            # g = utils.g(x_k).numpy()
            # g_grad = torch.autograd.functional.jacobian(utils.g, x_k.flatten(), create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            # constraints = [g_grad.numpy().T @ d + g <= 0]
            
            # QP SUB-PROBLEM
            problem = cp.Problem(objective, constraints)
            problem.solve()
            d_x = d.value
            d_λ = problem.constraints.dual_value
            
            # Update:  x_k+1 ← x_k + βdx
            x_k = x_k.numpy().flatten() + config.β * d_x
            
            # Update: λ_k+1 ← β*d_λ
            λ_k = config.β * d_λ

            # utils.plot_tensor(x_k, f"x_{k+1}")
            utils.show_image(x_k)
            print(f"Model prediction for x_{k+1}: {torch.argmax(model(x_k.reshape(1,28,28)))}")
                
    utils.show_image(x_k, title=f"x_{k}")

    
