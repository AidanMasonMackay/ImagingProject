import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

from functions import *

class FinDiff:
    """ Represents two-dimensional forward problem solver using finite differences method.
    Attributes
    ----------
    grid: dict
        grid dimension parameters

    nu: int
        source frequency
    Methods
    -------
    solve: Solves for the electric field inside the domain
    func: calculates objective function for input domain
    grad: calculates gradient of the objective function from input domain
    setFrequency: sets the frequency for the solver
    Notes
    -----
    """

    def __init__(self, grid, nu, P=12):
        # Grid params
        self.grid = grid
        self.x, self.y, self.delx  = grid['x'], grid['y'], grid['delx'] # grid dimensions in metres
        self.M, self.N, self.L, self.P = grid['M'], grid['N'], grid['L'], grid['P'] # Grid dimensions in cell numbers

        delx = self.delx
        M, N, L, P = self.M, self.N, self.L, self.P

        # Initialise PML
        sigma_opt = 0.7 * 0.02 / delx
        eps_max = 1

        # Sigma for PML in x and y directions
        sy = np.ones(M, dtype=complex)
        sy_half = np.ones(M + 1, dtype=complex)  # sigma halfway between gridpoints
        sx = np.ones(N, dtype=complex)
        sx_half = np.ones(N + 1, dtype=complex)   # sigma halfway between gridpoints

        ramp = np.zeros(P, dtype=complex) # Initialise sigma ramp-up in PML
        ramp_half = np.zeros(P, dtype=complex)

        # Construct PML
        for ii in range(0, P):
            ramp[ii] = ((P - (ii + 1)) / P) ** 3.7
            ramp_half[ii] = ((P - ((ii + 1) - 0.5)) / P) ** 3.7  # offset by 0.5

            sy[ii] = eps_max * ramp[ii] + 1 - complex(0, 1) * sigma_opt * ramp[ii]

            sy[M - ii - 1] = eps_max * ramp[ii] + 1 - complex(0, 1) * sigma_opt * ramp[ii]

            sy_half[ii] = eps_max * ramp_half[ii] + 1 - complex(0, 1) * sigma_opt * ramp_half[ii]
            sy_half[M - ii] = eps_max * ramp_half[ii] + 1 - complex(0, 1) * sigma_opt * ramp_half[ii]

            sx[ii] = eps_max * ramp[ii] + 1 - complex(0, 1) * sigma_opt * ramp[ii]
            sx[N - ii - 1] = sx[ii]

            sx_half[ii] = eps_max * ramp_half[ii] + 1 - complex(0, 1) * sigma_opt * ramp_half[ii]
            sx_half[N - ii] = sx_half[ii]

        # Save PML sigma values for forward prob solver
        self.sx_half = sx_half
        self.sx = sx
        self.sy_half = sy_half
        self.sy = sy

        # Set frequency
        self.nu = nu
        self.setFrequency(nu)

    def setFrequency(self, nu):
        self.nu = nu

        # Compute constants
        w = CalcAngularFrequency(nu)
        self.w = w
        ksquared = epsilon0 * mu0 * (2 * np.pi * nu) ** 2 # wave number squared
        self.ksquared = ksquared # Save to instance for forward prob solver

        sx_half = self.sx_half
        sx = self.sx
        sy_half = self.sy_half
        sy = self.sy
        N = self.N
        M = self.M
        L = self.L

        # helpful to have these as variables
        k1 = np.real(sx_half[1:N + 1]) + np.imag(sx_half[1:N + 1]) * (j / (2 * np.pi * nu * epsilon0))
        k2 = np.real(sx_half[0:N]) + np.imag(sx_half[0:N]) * (j / (2 * np.pi * nu * epsilon0))
        k = np.real(sx) + np.imag(sx) * (j / (2 * np.pi * nu * epsilon0))

        h = np.real(np.repeat(sy, N)) + np.imag(np.repeat(sy, N)) * (
                    j / (2 * np.pi * nu * epsilon0))  # y-derivatives need to be in blocks
        h1 = np.real(np.repeat(sy_half[1:], N)) + np.imag(np.repeat(sy_half[1:], N)) * (j / (2 * np.pi * nu * epsilon0))
        h2 = np.real(np.repeat(sy_half[0:-1], N)) + np.imag(np.repeat(sy_half[0:-1], N)) * (j / (2 * np.pi * nu * epsilon0))

        multiplier = 1 / (k1 * k2 * k)

        # partially fill in the main diagonal and all the -/+ x derivatives
        partial_x2 = sp.sparse.spdiags([k2, -(k1 + k2), k1], [-1, 0, 1], N, N).tocoo()  # more efficient to multiply from csr
        partial_x1 = (
            partial_x2.multiply((multiplier)[:, None])).tocsr()  # more efficient to create partial_x from sparse matrix

        # Save x and y partial derivatives for forward prob solver at frequency nu
        self.partial_x = sparse.csr_matrix(sp.sparse.kron(sp.eye(M), partial_x1))  # NB: assume zero-BCs on the outer
        self.partial_y = sp.sparse.spdiags([h2, -(h1 + h2), h1], [-N, 0, N], L, L).tocsr().multiply((1 / (h1 * h2 * h))[:, None])

    def solve(self, domain, excitation, waveguide = None):
        delx = self.delx
        ksquared = self.ksquared
        w = self.w
        L = self.L

        kappa = ConvertToKappa(domain, self.nu) # convert domain object to complex ndarray

        # Add waveguide to domain if passed as argument
        if waveguide:
            kappa += ConvertToKappa(waveguide, self.nu)

        # Solve forward problem
        partial_3 = delx ** 2 * ksquared * np.array(sp.sparse.spdiags(kappa, 0, L, L).tocsr()) # ksquared * kappa matrix
        A = self.partial_x + self.partial_y + partial_3  # A-matrix
        E = sp.sparse.linalg.spsolve(A, excitation)  # This is what takes all the time
        return E, A

    def func(self, kappa, Eobs, Pj_obs, excitation, waveguide):
        """"Objective function for a single source position"""
        Emod, _ = self.solve(kappa, excitation, waveguide)  # Solve for electric field
        # Calculate cost function corresponding to this source position
        J = 0.5 * np.sum((Eobs - Emod * Pj_obs) * ((Eobs - Emod * Pj_obs).transpose().conjugate()))
        return J

    def grad(self, kappa, Eobs, Pj_obs, excitation, waveguide):
        """"Gradient of objective function for a single source position"""
        delx = self.getdelx()
        ksquared = self.getksquared()
        Emod, A = self.solve(kappa, excitation, waveguide)
        a = Pj_obs * (Eobs - Pj_obs * Emod)  # Difference between model and observation
        lmult1 = sp.sparse.linalg.spsolve(A.transpose().conjugate(), a)  # solve the adjoint problem with Astar inc pml
        delL = np.real(ksquared * delx ** 2 * (Emod) * lmult1.transpose().conjugate())
        return delL

    def getFrequency(self):
        return self.nu

    def getdelx(self):
        return self.delx

    def getksquared(self):
        return self.ksquared

    def __str__(self):
        return str(self.x) + "x" + str(self.y) + "m grid with " + str(self.delx) + "m square cells finite differences solver"



class FinEl:
    def __init__(self):
        pass
    def solve(self):
        pass
    def func(self):
        pass
    def grad(self):
        pass
    def getFrequency(self):
        pass
    def getFrequency(self):
        pass
    def getdelx(self):
        pass
    def getksquared(self):
        pass