import numpy as np
import time
from OptimisationMethods.LineSearchMethods import *
from OptimisationMethods.NewtonTypeOptimisation import *
from scipy import optimize
import logging

def lbfgs(x0, func, grad, lineSearchMethod="pythonStrongWolfe", timeout = 60, max_iter=50, min_alpha=1e-15, tol = 1e-4, logger = None, options = {}):
    """Minimises cost function using L-BFGS algorithm
     Parameters
     ----------
    x0: ndarray, dtype = complex
        initial guess
    func: callable
        cost function
    grad: callable
        cost function gradient
    lineSearchMethod: str, optional
        name of line search method
    timeout: float, optional
        maximum run time
    max_iter: int, optional
        maximum number of iterations
    min_alpha: float, optional
        smallest accepted step size from line search
    tol: float, optional
        cost function tol
    options: dict, optional
        dictionary of line search options

     Returns
     -------
     res : OptimizeResult
         The optimization result
        """
    start = time.time() # record start time

    if not logger:
        logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler('LBFGS.log')
        formatter = logging.Formatter('%(asctime)s:%(message)s')
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)

    lineSearchMethods = ["Armijo", "strongWolfe", "pythonStrongWolfe"]
    if lineSearchMethod == "Armijo":
        lineSearch = Armijo_line_search
    elif lineSearchMethod == "strongWolfe":
        lineSearch = Strong_Wolfe
    elif lineSearchMethod == "pythonStrongWolfe":
        lineSearch = python_Strong_Wolfe
    else:
        raise Exception("line search method " + lineSearchMethod + " not recognised, choose from " + str(lineSearchMethods))

    # First iteration, k = 0
    xk = x0

    ndim = xk.shape[0]  # Get input dimensions
    I = np.eye(ndim)
    Hk = I  # Initial guess for inverse Hessian is the identity matrix
    delfk = grad(xk)

    pk = pk_bfgs(Hk, delfk)  # Solve for first search direction

    alphak = lineSearch(xk, pk, func, grad, **options)
    print(str(alphak))

    xk1 = xk + alphak * pk
    delfk1 = grad(xk1)
    sk = xk1 - xk
    yk = delfk1 - delfk

    # Initialise saved lists of y and s
    ss = np.array([sk])
    ys = np.array([yk])

    # Reset k
    xk = xk1
    delfk = delfk1

    k = 1  # Save iteration number

    tot_time = time.time() - start
    count = 0  # Count how many iterations are run through before time out
    while (func(xk) > tol and tot_time < timeout):  # While grad is more than tol and time is under chosen limit (seconds)
        pk = pk_lbfgs(Hk, delfk, ss, ys)
        alphak = lineSearch(xk, pk, func, grad, **options)

        xk1 = xk + alphak * pk
        costk1 = func(xk1)

        delfk1 = grad(xk1)
        sk = xk1 - xk
        yk = delfk1 - delfk

        # Add sk and yk to storage and drop (m+1)th terms - most recent at the front, least recent at the end
        if len(ss) < 10:
            ss = np.vstack((sk, ss))
            ys = np.vstack((yk, ys))
        else:
            ss = np.vstack((sk, ss[:-1]))
            ys = np.vstack((yk, ys[:-1]))

        # Reset k+1 --> k
        xk = xk1
        delfk = delfk1
        k += 1

        tot_time = np.round(time.time() - start, 2)

        # Different termination conditions
        if alphak < min_alpha:
            logger.debug("L-BFGS terminated because alpha <" + str(min_alpha))
            break
        if k >= max_iter:
            break
        if func(xk) < tol:
            break
        if tot_time > timeout:
            break
        count += 1
    logger.debug("test")
    return xk

def pk_bfgs(Hk, delfk):
    """ Computes search direction, pk, for use in BFGS algorithm """
    pk = -np.dot(Hk, delfk)
    return pk


def pk_lbfgs(H0kX, delfkX, ssX, ysX):
    """ Computes search direction, pk, for use in L-BFGS algorithm """

    # First loop, update q
    q = delfkX
    for i in np.arange(0, len(ssX)):  # From k-1 to k-m
        si = ssX[i]
        yi = ysX[i]
        rhoi = 1 / np.inner(yi, si)

        alphai = rhoi * np.inner(si, q)

    # Second loop, update r - find HkdelF on this iteration
    r = np.dot(H0kX, q)
    for i in np.flip(np.arange(0, len(ssX))):  # From k-m to k-1
        si = ssX[i]
        yi = ysX[i]
        rhoi = 1 / np.inner(yi, si)
        alphai = rhoi * np.inner(si, q)

        beta = rhoi * np.inner(yi, r)
        r = r + si * (alphai - beta)
    pk = -r
    return pk


def H0k1_lbfgs(skback, ykback, I):
    """" Approximate Hessian matrix for use in L-BFGS algorithm"""
    gammak = np.inner(skback, ykback) / np.inner(ykback, ykback)
    H0k1 = np.dot(gammak, I)
    return H0k1


def Hk1_bfgs(Hk, pk, sk, yk):
    """ Computes Hessian matrix for use in BFGS algorithm """
    b = I - pk * np.outer(sk, yk)  # Save b to make code more readable
    Hk1 = np.dot(np.dot(b, Hk), b) + pk * np.outer(sk, sk)
    return Hk1

def pythonlbfgs(x0, func, grad, options = {}):
    """ python algorithm as a reference """
    x = optimize.minimize(func, x0, jac=grad, method='l-bfgs-b', options = options)
    return x

def optimiseNewton(x0, func, grad, method = 'lbfgs', options = {}):
    """ interface for running Newton-type optimisation methods """
    methods = ['pythonlbfgs', 'lbfgs']
    if method == 'pythonlbfgs':
        x = pythonlbfgs(x0, func, grad, **options)
    elif method == 'lbfgs':
        x = lbfgs(x0, func, grad, **options)
    else:
        raise Exception("Newton-type optimisation method not recoginised, choose from " + methods[1:-1])
    return x