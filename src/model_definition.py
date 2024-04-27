import torch
import cvxpy as cp
import time

def f_function(x, xk):
    return (1/2) * torch.norm(x - xk, p='fro') ** 2

def define_optimization_variables(dim):
    d = cp.Variable(dim)
    I = torch.eye(dim)
    return d, I
 
def compute_gradients(x, xk):
    gradient = torch.norm(torch.flatten(x) - torch.flatten(xk))
    return gradient

def setup_constraints(g, xk):
    g_val = cp.Variable(g(xk).shape, value=g(xk).detach().numpy())
    return [g_val <= 0]

def perform_optimization(f, x, xk, d, I, alpha, constraints):
    f_val = f(x, xk)
    f_gradient = compute_gradients(x, xk)
    objective = f_gradient.T @ d + 0.5 * cp.quad_form(d, I)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    result = problem.solve()
    optimal_d = d.value
    return xk.flatten() + alpha * optimal_d