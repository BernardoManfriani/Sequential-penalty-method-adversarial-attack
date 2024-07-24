import torch
import os
import sys
import numpy as np
import cvxpy as cp
from src import data_preparation
import matplotlib.pyplot as plt

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

def C(x):
    with torch.no_grad():
        x = x.to(device)
        x_reshaped = x.view(1, 1, 28, 28)
        return model(x_reshaped).cpu().numpy()
    
def SQUAT(x, N_iter, N_1, α, β):
    
    j = config.target_class
    n = 28 * 28
    K = config.classes
    λ_k = np.zeros(10)  # Lagrange multipliers
    target_logits = []
    original_logits = []
    iterations = []
    
    x_k = utils.get_random_image(j, dataset, seed=123)
    # utils.show_image(x, title="Original image")
    initial_logits = C(x)
    original_class = np.argmax(initial_logits)
    print(f"Classificazione iniziale: {original_class}")
    # utils.show_image(x_k, title="x0")
    
    
    for k in range(N_iter):
        
        # FIRST ORDER
        '''
        min_d ∇f(xk)^T*d + 0.5*d^T*I*d
        ∇g(xk)^T*d + g(xk) ≤ 0 
        '''
        if k < N_1:
            print(f"\nIterazione {k+1}/{config.N_iter} (first order)")
            
            # f_grad = utils.f_gradient(x, x_k)
            f_grad = (x_k - x).flatten().cpu().numpy()
            C_x_k = C(x_k)
            g_x_k = (np.eye(K) - np.ones((K, 1)) @ np.eye(1, K, j)) @ C_x_k.T
            g_grad = np.zeros((K, 784)) 
            epsilon = 1e-5
            
            ''' g'(xk) i-esima = g(x_k+e_i*epsilon)-g(x_k)/epsilon '''
            for i in range(784):
                x_plus = x_k.clone()
                x_plus.flatten()[i] += epsilon
                C_x_plus = C(x_plus)
                g_x_plus = (np.eye(K) - np.ones((K, 1)) @ np.eye(1, K, j)) @ C_x_plus.T
                g_grad[:, i] = (g_x_plus - g_x_k).flatten() / epsilon      
            
            # g_x_k = utils.g(x_k)
            # g_grad = torch.autograd.functional.jacobian(utils.g, x_k).numpy() # (10, 784)
            
            d = cp.Variable(n)  
            objective = cp.Minimize(f_grad.T @ d + 0.5 * cp.quad_form(d, np.eye(n))) # objective = cp.Minimize(cp.sum_squares(x_k.flatten()-x.flatten()+d))
            constraints = [g_grad @ d + g_x_k.flatten() <= 0] # (10, 784)@(784,) + (10)

            
            # QUADRATIC SUB-PROBLEM
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=False) # solver=cp.OSQP, cp.ECOS, cp.GLPK
            optimal_d = torch.from_numpy(d.value).cpu()
            
            # UPDATE
            # x_k = x_k + config.α * torch.from_numpy(d.value).float().to(device).view(1, 28, 28)
            x_k = x_k + α * optimal_d.float().view(1, 28, 28)
            x_k = x_k.view(1, 28, 28).cpu()
            utils.show_image(x_k, f"x_{k}")
            # x_k = x_k.clamp(min=0.0, max=1.0).reshape(28, 28).to(torch.float32)
            
            current_logits = C(x_k)
            target_logits.append(current_logits[0, j])
            original_logits.append(current_logits[0, original_class])
            iterations.append(k+1)
            
            print(f"Model prediction for x_{k+1}: {np.argmax(current_logits)}")
            
            if config.verbose:
                if (k+1) % 10 == 0:
                    utils.show_image(x_k, f"Immagine perturbata (Iterazione {k+1})")
                    print(f"logits: {current_logits}")
                    print(f"||x - x_k||: {torch.norm(x.flatten() - x_k.flatten(), p='fro')}")
                    plt.figure(figsize=(10, 6))
                    plt.plot(iterations, target_logits, label=f'Classe target ({j})')
                    plt.plot(iterations, original_logits, label=f'Classe originale ({original_class})')
                    plt.xlabel('Iterazioni')
                    plt.ylabel('Logits')
                    plt.title('Evoluzione dei logits durante la perturbazione di primo ordine')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                
            if np.argmax(current_logits) == j:
                print("SmallCNN has been corrupted")
        
        # # SECOND ORDER
        # else:
        #     print(f"\nIterazione {k+1}/{config.N_iter} (second order)")
            
        #     f_grad = utils.f_gradient(x, x_k)
            
        #     L_hessian = torch.autograd.functional.hessian(utils.lagrangian, inputs=(x, x_k, torch.from_numpy(λ_k)))[1][1]
        #     L_hessian = L_hessian.view(784, 784) # ; print(L_hessian)
        
        #     objective = cp.Minimize(f_grad.numpy().T @ d + 0.5 * (d @ L_hessian @ d.T))
        #     # objective = cp.Minimize(f_grad.numpy().T @ d + 0.5 * cp.quad_form(d, L_hessian))
            
        #     # CONSTRAINTS
        #     g = cp.Variable(utils.g(x_k).shape, value=utils.g(x_k).numpy())
        #     g_grad = torch.autograd.functional.jacobian(utils.g, x_k.flatten()).numpy()
        #     constraints = [g_grad @ d + g <= 0]
            
        #     # QP SUB-PROBLEM
        #     problem = cp.Problem(objective, constraints)
        #     result = problem.solve(verbose=True)
        #     # result = problem.solve(solver=cp.OSQP, verbose=True)
        #     # result = problem.solve(solver=cp.ECOS, verbose=True)
        #     # result = problem.solve(solver=cp.GLPK, verbose=True)
        #     d_x = d.value # problem solution
        #     d_λ = constraints[0].dual_value # ; print(f"d_λ: {d_λ}") # lagrangian multipliers
            
        #     # Update x_k and λ_k
        #     x_k = x_k.flatten() + config.β * d_x
        #     x_k = x_k.clamp(min=0.0, max=1.0).reshape(28, 28).to(torch.float32)
        #     λ_k = config.β * d_λ # ; print(f"λ_k: {λ_k}")
            
        #     logits = model(x_k.unsqueeze(0))
        #     if torch.argmax(logits) == j:
        #         print("SmallCNN has been corrupted")
                
        #     if config.verbose == True:
        #         print(f"||x - x_k||: {torch.norm(x.flatten() - x_k.flatten(), p='fro')}")
            
                
        #         print(f"logits: {logits.detach().numpy()}")
        #         print(f"Model prediction for x_{k+1}: {torch.argmax(logits)}")
               
        #         # utils.show_image(x_k, title=f"x{k+1}")
        #         utils.plot_tensor(x_k, title=f"Second order x{k+1}")
           

    utils.show_image(x_k, title=f"x_{k}")

