
from OptimisationMethods.AEI.DiscretisationMethods import FinDiff, FinEl
from OptimisationMethods.NewtonTypeOptimisation import optimiseNewton
from ObjectiveFunction import buildFuncAndGrad
from functions import ConvertKappaToDict, ConvertToKappa, subtractDomains, addDomains
from visualise import plot_kappa

import numpy as np

def AEI(x_init, x_background, experimental_data, optimisation_grid, num_eigenfunctions_list, trunc_dist = 0, newton_method ='lbfgs', newton_options = {}, linesearch_options = {}, convert_laplacian = True):
    """Minimises cost function using Adaptive Eigenspace Inversion
     Parameters
     ----------
    x_init: ndarray, dtype = complex
        initial guess
    x_background: ndarray, dtype = complex
        'background' value. Should be chosen using a priori knowledge about the parameter
    experimental_data: dict
        contains electric field measurements from either real or simulated data, and details of the experimental set-up (see ExperimentalSetup.ExperimentBuilder class for more details)
    optimisation_grid: dict
        contains grid dimension parameters and discretisation method to be used during optimisation. Note this is different to simulation_grid, which is the grid the E-field measurements were simulated on (if using simulated data)
    num_eigenfunctions_list: list
        number of eigenvectors to be used at each stage of the AEI algorithm. len(numEigenvecsList) determines the number of basis adaptions to be performed
    trunc_dist: float, optional
        truncate optimisation space to cells which are more than trunc_dist metres from the domain boundary
    newton_method: str, optional
        name of Newton type algorithm to be used
    convertLaplacian: bool, optional
        if True, initial guess is converted in and out of the first 150 eigenvectors of the Laplacian before the AEI algorithm is run. This addresses problems that arise when calculating the eigenvectors of a homogeneous array

     Returns
     -------
     res : ndarray, dtype = real (note --- this is planned to be extended to complex arrays in future for multi-parameter reconstructions)
         The optimization result
        """

    # Create methods for calculating objective grad and func
    func, grad = buildFuncAndGrad(experimental_data, optimisation_grid)

    # Create class instance for calculating eigenfunctions
    nu = experimental_data['nu']
    methods = ['FinDiff', 'FinEl']
    if optimisation_grid['discretisationMethod'] == 'FinDiff':
        AEImethod = FinDiff(optimisation_grid, nu, trunc_dist)
    elif optimisation_grid['discretisationMethod'] == 'FinEl':
        AEImethod = FinEl(optimisation_grid, nu, trunc_dist)
    else:
        raise Exception("discretisation method '" + optimisation_grid['discretisationMethod'] + "' not recognised. Must be one of " + str(methods)[1:-1])

    # Prepare x for optimisation algorithm
    if convert_laplacian == True:
        _, Lvectors = AEImethod.calcLaplaceEigenfunctions(num_eigenfunctions=150)
        CtoE, EtoC = basis_conversions(Lvectors) # methods for conversions between cartesian basis and eigenfunctions of Laplacian
        x_init_converted = EtoC(CtoE(x_init))
        x = subtractDomains(x_init_converted.copy(), x_background)  # remove background from initial guess
    else:
        x = subtractDomains(x_init.copy(), x_background)  # remove background from initial guess

    # First iteration of AEI algorithm is performed using eigenfunctions of the Laplacian operator
    _, vectors = AEImethod.calcLaplaceEigenfunctions(num_eigenfunctions=num_eigenfunctions_list[0])

    CtoE, EtoC = basis_conversions(vectors)  # methods for conversion between eigenbasis and cartesian basis
    funcAEI, gradAEI = buildAEIobjective(CtoE, EtoC, func, grad)  # create methods for calculating objective function and its gradient

    # Run Newton type optimisation
    xEig = CtoE(x)  # Convert guess x into eigenbasis

    # need to re-think how kappa is saved. May be better as an array, not a dict
    nu_placeholder = 1e9 # this could be anything. Needed for converting kappa between dict and numpy array
    xnewEig = optimiseNewton(ConvertToKappa(xEig, nu = nu_placeholder), funcAEI, gradAEI, options=linesearch_options, **newton_options)

    xnew = EtoC(ConvertKappaToDict(xnewEig, nu = nu_placeholder))
    x = xnew  # update x

    plot_kappa(optimisation_grid, x, nu)  ### TEMP

    for num_eigenfunctions in num_eigenfunctions_list[1:]:
        # Calculate eigenvectors and values in truncated domain
        _, vectors = AEImethod.calcEigenFunctions(x, num_eigenfunctions=num_eigenfunctions)

        CtoE, EtoC = basis_conversions(vectors) # methods for conversion between eigenbasis and cartesian basis
        funcAEI, gradAEI = buildAEIobjective(CtoE, EtoC, func, grad) # create methods for calculating objective function and its gradient

        # Run Newton type optimisation
        xEig = CtoE(x) # Convert guess x into eigenbasis
        xnewEig = optimiseNewton(ConvertToKappa(xEig, nu=nu_placeholder), funcAEI, gradAEI, options=linesearch_options,
                                 **newton_options)

        xnew = EtoC(ConvertKappaToDict(xnewEig, nu=nu_placeholder))
        x = xnew.copy() # update x
        plot_kappa(optimisation_grid, x, nu)  ### TEMP

    xfinal = addDomains(xnew, x_background) # add background back into solution
    return xfinal

def buildAEIobjective(CtoE, EtoC, func, grad):
    """ Modifies objective func and grad functions to include basis conversion """
    def funcAEI(kappaEig):
        """ Convert kappa into a cartesian basis, then calculate objective function"""
        kappaCart = EtoC(kappaEig)
        J = func(kappaCart)
        return J

    def gradAEI(kappaEig):
        """ Convert kappa into a cartesian basis, then calculate gradient of objective function"""
        kappaCart = EtoC(kappaEig)
        gradientCart = grad(kappaCart)
        gradientEig = CtoE(gradientCart)
        return gradientEig

    return funcAEI, gradAEI

def basis_conversions(vecs):
    """ Create functions for performing basis changes between cartesian coordinates and eigenbasis """
    def CtoE(kappa_cart):
        if type(kappa_cart) == dict: # if dict, split into eps and sigma first
            kappa_eig_eps = np.matmul(vecs.T, kappa_cart['eps'])
            kappa_eig_sigma = np.matmul(vecs.T, kappa_cart['sigma'])
            kappa_eig = {'eps': kappa_eig_eps, 'sigma': kappa_eig_sigma}
        else:
            kappa_eig = np.matmul(vecs.T, kappa_cart)
        return kappa_eig

    def EtoC(kappa_eig):
        if type(kappa_eig) == dict: # if dict, split into eps and sigma first
            kappa_cart_eps = np.matmul(kappa_eig['eps'], vecs.T)
            kappa_cart_sigma = np.matmul(kappa_eig['sigma'], vecs.T)
            kappa_cart = {'eps': kappa_cart_eps, 'sigma': kappa_cart_sigma}
        else:
            kappa_cart = np.matmul(kappa_eig, vecs.T)
        return kappa_cart
    return CtoE, EtoC