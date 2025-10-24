
# Package setup, assuming Python was installed using Anaconda:
# Place this script in a folder called "comp_econ". Then place the comp_econ folder into site-packages folder
# of desired environment.

import numpy as np

# Googled "gauss jacobi and gauss seidel method python" to find gjacobi and gseidel functions.
# Tested several matrices and obtained same results as when using gjacobi and gseidel methods
# from CompEcon Toolbox for MATLAB.

def gjacobi(A, b, x0, tol=1e-6, max_iter=1000):
    """If the Gauss-Jacobi method converges for the passed arguments,
    then the solution vector x and the number of iterations are returned.
    Otherwise, x is returned at the max_iter number of iterations."""

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
            print(f"Gauss-Jacobi method converged in {k+1} iterations.")
            return x_new, k+1
        
        x = np.copy(x_new)
        
    print(f"Gauss-Jacobi method did not converge within {max_iter} iterations.")
    return x

def gseidel(A, b, x0, tol=1e-6, max_iter=1000):
    """If the Gauss-Seidel method converges for the passed arguments,
    then the solution vector x and the number of iterations are returned.
    Otherwise, x is returned at the max_iter number of iterations."""

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
            return x, k+1
            
    print(f"Gauss-Seidel method did not converge within {max_iter} iterations.")
    return x

def bisect(f, a, b, tol=1e-6, max_iter=100, args=()):
    """
    Bisection root-finding algorithm with iteration counter.

    Parameters:
    -----------
    f : callable
        Function of a single variable, optionally with extra args.
    a, b : float
        Interval [a, b] where f changes sign.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum number of iterations.
    args : tuple
        Extra arguments to pass to f.

    Returns:
    --------
    root : float
        Approximate root of f(x) in [a, b].
    iterations : int
        Number of iterations performed.
    """
    fa = f(a, *args)
    fb = f(b, *args)
    if fa * fb > 0:
        raise ValueError("Function does not change sign on [a, b].")

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c, *args)

        if abs(fc) < tol or (b - a)/2 < tol:
            return c, i  # Return root and iteration count

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    raise RuntimeError("Bisection method did not converge within max_iter.")
