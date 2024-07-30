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
    
def compute_g_gradient(x_k, j, K):
    x_k = x_k.clone().detach().requires_grad_(True)
    
    def g(x):
        C_x = C(x)  # Assumendo che model sia il tuo modello PyTorch
        return torch.from_numpy((np.eye(K) - np.ones((K, 1)) @ np.eye(1, K, j)) @ C_x.T)
    
    # Calcola lo Jacobian
    jacobian = torch.autograd.functional.jacobian(g, x_k)
    
    # Lo Jacobian avrà dimensioni (K, 1, 28, 28) se x_k è (1, 28, 28)
    # Rimodelliamo per ottenere (K, 784)
    g_grad = jacobian.view(K, -1)
    
    return g_grad.detach().numpy()



def SQUAT(x, x_j, j, N_iter, N_1, α, β):
    # riceve un immagine appartente alla classe target (rimane costante)
    # aggiorna xk che all'inizio è un numero diverso dalla classe target
    n = 28 * 28
    K = config.classes
    λ_k = np.zeros(10)  # Lagrange multipliers
    target_logits = []
    original_logits = []
    iterations = []
    norms = []
      
    x_k = x_j
    
    utils.show_image(x, title="Image to attack")
    initial_logits = C(x)
    original_class = np.argmax(initial_logits)
    print(f"Classificazione iniziale: {original_class}")
    utils.show_image(x_k, title="x0")
    
    
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
            
            ''' g_grad con differenze finite  (g'(xk) i-esima = g(x_k+e_i*epsilon)-g(x_k)/epsilon) '''
            g_grad = np.zeros((K, 784)) 
            epsilon = 1e-5
<<<<<<< Updated upstream
=======
            
            #differenze finite per calcolare il gradiente di g
            ''' g'(xk) i-esima = g(x_k+e_i*epsilon)-g(x_k)/epsilon '''
>>>>>>> Stashed changes
            for i in range(784):
                x_plus = x_k.clone()
                x_plus.flatten()[i] += epsilon
                C_x_plus = C(x_plus)
                g_x_plus = (np.eye(K) - np.ones((K, 1)) @ np.eye(1, K, j)) @ C_x_plus.T
                g_grad[:, i] = (g_x_plus - g_x_k).flatten() / epsilon  
            print(f"x.device: {x.device}")    
            print(f"xk.device: {x_k.device}")    
            
            '''g_grad con jacobiano'''
            g_grad = compute_g_gradient(x_k, j, K)

            d = cp.Variable(n)  
            objective = cp.Minimize(f_grad.T @ d + 0.5 * cp.quad_form(d, np.eye(n))) # objective = cp.Minimize(cp.sum_squares(x_k.flatten()-x.flatten()+d))
            constraints = [g_grad @ d + g_x_k.flatten() <= 0] # (10, 784)@(784,) + (10)

            
            # QUADRATIC SUB-PROBLEM
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=False) # solver=cp.OSQP, cp.ECOS, cp.GLPK
            
            if d.value is None:
                optimal_d = torch.rand(n)
            else:    
                # optimal_d = torch.from_numpy(d.value).clamp(0,1).cpu()
                optimal_d = torch.from_numpy(d.value).cpu()
            
            # print(optimal_d)
            
                
            # UPDATE
            x_k = x_k + α * optimal_d.float().view(1, 28, 28)
            # x_k = x_k + α * optimal_d.float().view(1, 28, 28).clamp(0,1)
            x_k = x_k.view(1, 28, 28).cpu()
            
            current_logits = C(x_k)
            target_logits.append(current_logits[0, j])
            original_logits.append(current_logits[0, original_class])
            iterations.append(k+1)
            norms.append(torch.norm(x.flatten() - x_k.flatten(), p='fro'))
            print(f"Model prediction for x_{k+1}: {np.argmax(current_logits)}")
            
            if config.verbose:
                if (k+1) ==1:
                    utils.show_image(x_k, f"Immagine perturbata (Iterazione {k+1})")
                    print(f"logits: {current_logits}")
                    print(f"||x - x_k||: {torch.norm(x.flatten() - x_k.flatten(), p='fro')}")
                    
                if (k+1) ==5:
                    utils.show_image(x_k, f"Immagine perturbata (Iterazione {k+1})")
                    print(f"logits: {current_logits}")
                    print(f"||x - x_k||: {torch.norm(x.flatten() - x_k.flatten(), p='fro')}")
                if (k+1) ==50:
                    utils.show_image(x_k, f"Immagine perturbata (Iterazione {k+1})")
                    print(f"logits: {current_logits}")
                    print(f"||x - x_k||: {torch.norm(x.flatten() - x_k.flatten(), p='fro')}")
                        
                if (k+1) % 50 == 0:
                    plt.figure(figsize=(10, 6))
                    plt.plot(iterations, target_logits, label=f'Classe target ({j})')
                    plt.plot(iterations, original_logits, label=f'Classe originale ({original_class})')
                    plt.xlabel('Iterazioni')
                    plt.ylabel('Logits')
                    plt.title('Evoluzione dei logits durante la perturbazione di primo ordine')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                    
                if (k+1) % 50 == 0:
                    plt.figure(figsize=(10, 6))
                    plt.plot(norms, label='||x - x_k||')
                    plt.ylabel('Norma di Frobenius')
                    plt.xlabel('Iterazioni')
                    plt.title('Evoluzione della norma di Frobenius')
                    plt.legend()
                    plt.grid(True)
                    plt.show()
                
            if np.argmax(current_logits) == j:
                print("SmallCNN has been corrupted")
        
        
        if k > N_1:
            def g(x):
                C_x = C(x) 
                return torch.from_numpy((np.eye(K) - np.ones((K, 1)) @ np.eye(1, K, j)) @ C_x.T)
            
            f = 0.5*torch.norm(x_k - x)**2
            def L(x):
                print(f"f.type: {f.type}")
                return f + torch.sum(λ_k * g(x))
            
            # SECOND ORDER
            ''' 
                min_d ∇f(xk)^T*d + 0.5*d^T*∇^2L_xx(x_k, λ_k)*d 
                ∇g(xk)^T*d + g(xk) ≤ 0 
            '''
            print(f"\nIterazione {k+1}/{config.N_iter} (second order)")
            
            f_grad = (x_k - x).flatten().cpu().numpy()
            C_x_k = C(x_k)
            g_x_k = (np.eye(K) - np.ones((K, 1)) @ np.eye(1, K, j)) @ C_x_k.T
            
            H = torch.autograd.functional.hessian(L, x_k)
            H = H.detach().numpy().reshape(784, 784)
            
            d = cp.Variable(n)
            objective = cp.Minimize(f_grad.T @ d + 0.5 * cp.quad_form(d, H))
            constraints = [g_grad @ d + g_x_k <= 0]
            
            problem = cp.Problem(objective, constraints)
            problem.solve(verbose=False)
            
            # Aggiornamento di x_k
            d_x = d.value
            d_λ = constraints[0].dual_value # ; print(f"d_λ: {d_λ}")
            
            x_k_new = x_k.detach().numpy().flatten() + β * d_x
            # x_k = torch.tensor(x_k_new.reshape(1, 28, 28), dtype=torch.float32)
            # x_k = x_k.clamp(min=0.0, max=1.0).reshape(28, 28).to(torch.float32)
            λ_k = β * d_λ # ; print(f"λ_k: {λ_k}")
        
    utils.show_image(x_k, title=f"x_{k}")

