
# Package setup, assuming Python was installed using Anaconda:
# Place this script in a folder called "comp_econ". Then place the comp_econ folder into site-packages folder
# of desired environment.

import numpy as np

# Googled "gauss jacobi and gauss seidel method python" to find gjacobi and gseidel functions.
# Tested several matrices and obtained same results as when using gjacobi and gseidel methods
# from CompEcon Toolbox for MATLAB.

def gjacobi(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.copy(x0)
    x_new = np.zeros_like(x)
    
    for k in range(max_iter):
        for i in range(n):
            sum_val = 0
            for j in range(n):
                if i != j:
                    sum_val += A[i, j] * x[j]
            x_new[i] = (b[i] - sum_val) / A[i, i]
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Jacobi method converged in {k+1} iterations.")
            return x_new
        
        x = np.copy(x_new)
        
    print(f"Jacobi method did not converge within {max_iter} iterations.")
    return x

def gseidel(A, b, x0, tol=1e-6, max_iter=1000):
    n = len(b)
    x = np.copy(x0)
    
    for k in range(max_iter):
        x_old = np.copy(x)
        for i in range(n):
            sum_val = 0
            for j in range(n):
                if i != j:
                    sum_val += A[i, j] * x[j]
            x[i] = (b[i] - sum_val) / A[i, i]
        
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            print(f"Gauss-Seidel method converged in {k+1} iterations.")
            return x
            
    print(f"Gauss-Seidel method did not converge within {max_iter} iterations.")
    return x