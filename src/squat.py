import torch
import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cvxpy as cp
from src import utility_functions
import torchvision
from models.small_cnn import SmallCNN
from tqdm import tqdm


j = 1 # target class
N_iter = 12 # total number of iterations
N_1 = 10 # number of the first order iterations
alpha = 1 # learning rate of the first order iterations
beta = 0.15 # learning rate of the second order iterations
K = 28*28
f_gradient = torch.zeros(28*28)

model = SmallCNN()
# model.load_state_dict(torch.load("training/smallcnn_regular/model-nn-epoch10.pt")) #For CUDA
model.load_state_dict(torch.load("training/smallcnn_regular/model-nn-epoch10.pt", map_location=torch.device('cpu')))

def squat_algorithm(x, xk):
    for k in range (0, N_iter):
        # FIRST ORDER
        if k<=N_1:
            d = cp.Variable(28*28) # direction (variable of the optimization problem)
            I = torch.eye(28*28) # identity matrix
            f_val = utility_functions.f(x, xk) # frobenius_norm between x and xk

            for i in range(28*28):
                f_gradient[i] = torch.norm(torch.flatten(x)[i]-torch.flatten(xk)[i]) # gradient of f(x) (vector in R^K)

            objective = f_gradient.t()@d + 0.5 * cp.quad_form(d, I)

            # CONSTRAINTS - finire
            g_val = cp.Variable(utility_functions.g(xk,K).shape, value=g(xk,K).detach().numpy())
            jacobian = torch.autograd.functional.jacobian(utility_functions.g, xk, create_graph=False, strict=False, vectorize=False, strategy='reverse-mode')
            constraints = [g_val <= 0]

            # QP PROBLEM
            problem = cp.Problem(cp.Minimize(objective), constraints) # definition of the constrained problem to minimize
            result = problem.solve()
            optimal_d = d.value
            # print(f"d.value: {d.value}")

            # UPDATE
            xk = xk.flatten() + alpha * optimal_d
            xk = xk.reshape(28,28)
            xk = xk.to(torch.float32)
            utility_functions.plot_tensor(xk, f"x{k+1}", dim=2)
            print(f"Model prediction for x_{k+1}: {torch.argmax(model(xk.reshape(1,28,28)))}")

        # SECOND ORDER
        # torch.autograd.functional.hessian or torch.autograd.functional.jacobian
        # end_time = time.time()
        # print(f"Iteration time: {end_time - start_time:.10f} sec.")
