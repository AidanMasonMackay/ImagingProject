
import numpy as np

# Physical constants
j = complex(0, 1)
epsilon0 = 8.85e-12
mu0 = 4 * np.pi * 1e-7
c0 = 1 / np.sqrt(epsilon0 * mu0)

def CalcAngularFrequency(nu):
    """ convert frequency (Hz) to angular frequanecy (rad s^-1)"""
    return np.pi * 2 * nu # angular frequency

def MetresToCells(x, delx): # This is particular to finite differences. Should be moved
    """ convert distance in metres to approximate number of cells """
    NumCells = x/delx
    return NumCells

def ConvertKappaToDict(domain, nu):
    """ Builds dictionary object of kappa. Useful for saving and loading kappa, and for decoupling eps and sigma from the frequency used in experiment """
    if type(domain) == dict: # in case input was already a dictionary, return input
        return domain
    else:
        w = np.pi * 2 * nu  # angular frequency

        eps = np.real(domain)
        kappaimag = np.imag(domain)

        sigma = kappaimag*w*epsilon0
        return {'eps': eps, 'sigma':sigma}

def ConvertToKappa(domain, nu):
    """calculates relative wave number for a given eps and sigma and returns kappa as a numpy array"""
    if type(domain) == dict:
        w = np.pi * 2 * nu  # angular frequency
        eps, sigma = np.array(domain['eps']), np.array(domain['sigma'])
        kappa = eps + ((1 / epsilon0) * (j * sigma) / w)
        return kappa
    else: # In case input was already an array, return input
        return domain
def addDomains(domain1, domain2):
    """ adds the eps and sigma values for two domains. Useful for adding diseases to healthy trees, or adding background kappa in AEI """
    domain = {}
    domain['eps'] = domain1['eps']+domain2['eps']
    domain['sigma'] = domain1['sigma']+domain2['sigma']
    return domain

def subtractDomains(domain1, domain2):
    """ subtracts the eps and sigma values in domain2 from domain1. Useful for removing background kappa in AEI """
    domain = {}
    domain['eps'] = domain1['eps']-domain2['eps']
    domain['sigma'] = domain1['sigma']-domain2['sigma']
    return domain

def BuildFinDiffGrid(x, y, delx, P=12):
    """
    Constructs grid dimension parameters for finite differences method
    Parameters
    ----------
    x : float
        domain size in x direction (metres).
    y : float
        domain size y direction (metres)
    delx: float
        grid cell width and height (metres)
    """
    M = int(x / delx)  # num cells in Y
    N = int(y / delx)  # num cells in X
    L = int(N * M)  # total num cells

    grid = {'M': M, 'N': N, 'L': L, 'x': x, 'y': y, 'delx': delx, 'P': P, 'discretisationMethod':'FinDiff'}
    return grid

def BuildFinElGrid():
    pass