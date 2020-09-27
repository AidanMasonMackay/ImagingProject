
import numpy as np
from scipy.optimize import line_search

def Armijo_line_search(xk, pk, func, grad, alpha0=0.9, tau=0.9, c1=1e-4):
    """" Armijo line search algorithm """
    alphaj = alpha0
    m_line = np.inner(grad(xk), pk)  # approximate step size for the change in the function
    j = 0
    t = -c1 * m_line
    while 1 == 1:
        diff = func(xk) - func(xk + alphaj * pk)  # Difference between function before and after the step
        if diff <= alphaj * t:  # If doesn't meet sufficient decrease conditions, keep going
            alphaj = tau * alphaj  # Decrease alphaj by factor, tau
            j += 1
        else:
            return alphaj

def python_Strong_Wolfe(xk, pk, func, grad, c1=1e-4, c2=0.1):
    """" SciPy implementation of strong Wolfe line search algorithm """
    output = line_search(func, grad, xk, pk, gfk=None, old_fval=None, old_old_fval=None, args=(), c1=c1, c2=c2, amax=None, extra_condition=None, maxiter=10)
    return output[0]

def Strong_Wolfe(xk, pk, func, grad, c1=1e-4, c2=0.1, alphamax=16, delalpha=1, max_iterations=100):
    """ Self-built strong Wolfe line search algorithm """
    alphak = bracket(xk, pk, func, grad, c1, c2, alphamax, delalpha, max_iterations)
    return alphak

# Strong Wolfe Conditions line search bracketing and zoom phases
def bracket(xk, pk, func, grad, c1, c2, alphamax, delalpha, max_iterations):
    alphaiminus1 = 0  # Initialise alphai-1
    func0 = func(xk)  # func at alpha = 0
    grad0 = np.inner(grad(xk), pk)  # directional derivative at alpha = 0
    funciminus1 = func0  # func at alpha = alphai-1
    alphai = delalpha  # First test alpha - Nodedal book says delalpha = 1 is a good choice for quasi-Newton methods

    # loop to find (alphalow, alphahigh) that meets bracket conditions
    count = 0
    while count <= max_iterations:
        funci = func(xk + alphai * pk)  # func at alpha = alphai
        gradi = np.inner(grad(xk + alphai * pk), pk)  # directional derivative at alpha = alphai

        ## Return alphastar if any termination conditions met -->

        # Sufficient decrease conditions met or the cost function increased
        if ((funci > (func0 + c1 * alphai * grad0)) or (funci >= funciminus1 and alphaiminus1 != 0)):
            alphalow, alphahigh = alphaiminus1, alphai
            alphastar = zoom(xk, pk, func, grad, alphalow, alphahigh, c1, c2, max_iterations)
            return alphastar

        # Curvature conditions met
        if (np.abs(gradi) < -c2 * grad0):
            alphastar = alphai
            return alphastar

        # Gradient is positive
        if (gradi >= 0):
            alphalow, alphahigh = alphaiminus1, alphai
            alphastar = zoom(xk, pk, func, grad, alphalow, alphahigh, c1, c2, max_iterations)
            return alphastar

        # If no termination condition met, increase alpha and test again
        alphaiminus1 = alphai
        alphai += delalpha
        count += 1


def zoom(xk, pk, func, grad, alphalow, alphahigh, c1, c2, max_iterations):
    func0 = func(xk)  # func at alpha = 0
    grad0 = np.inner(grad(xk), pk)  # directional derivative at alpha = 0
    count = 0

    # loop to find alphastar in (alphahigh, alphalow) that meets strong Wolfe conditions
    while count <= max_iterations:
        phi_lo = func(xk + pk * alphalow) # phi corresponding to alphalow
        phi_hi = func(xk + pk * alphahigh) # phi corresponding to alphahigh
        derphi_lo = np.inner(grad(xk + pk * alphalow), pk) # directional derivative of phi corresponding to alphalow

        alphaj = _quadmin(alphalow, phi_lo, derphi_lo, alphahigh, phi_hi)  # Quadratic interpolation to choose initial trial alphaj
        funcj = func(xk + alphaj * pk)  # func at alpha = alphaj
        funcalphalow = func(xk + alphalow * pk)  # func at alpha = alphalow

        # Check if alphaj violates sufficient decrease conditions or the cost function is higher than at the alpha lower bound
        if ((funcj > func0 + c1 * alphaj * grad0) or (funcj >= funcalphalow)):
            alphahigh = alphaj  # alphaj is now the upper bound of the search bracket
        else:
            gradj = np.inner(grad(xk + pk * alphaj), pk)  # directional derivative at alpha

            # Accept alphaj if meets strong Wolfe conditions
            if np.abs(gradj) <= -c2 * grad0:
                alphastar = alphaj
                return alphastar

            # If the estimated cost function over this region is positive, move the bracket higher
            if gradj * (alphahigh - alphalow) >= 0:
                alphalow = alphahigh

            # Increase lower limit of the bracket
            alphalow = alphaj
        count += 1

def _quadmin(a, fa, fpa, b, fb):
    """ from https://github.com/scipy/scipy/blob/adc4f4f7bab120ccfab9383aba272954a0a12fb0/scipy/optimize/linesearch.py#L526 """
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa,
    """
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin