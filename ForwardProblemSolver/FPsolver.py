
from ForwardProblemSolver.DiscretisationMethods import FinDiff, FinEl

class FPsolver:
    """ Represents two-dimensional forward problem solver for any discretisation method.
    Attributes
    ----------
    grid: dict
        grid dimension parameters and discretisation method
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
    def __init__(self, grid, nu):
        discretisationMethod = grid['discretisationMethod']
        if discretisationMethod == 'FinDiff':
            self.solver = FinDiff(grid, nu) # Create an instance of discretisation method
        if discretisationMethod == 'FinEl':
            self.solver = FinEl(grid, nu) # Create an instance of discretisation method

    def solve(self, domain, excitation, waveguide = None):
        E, A = self.solver.solve(domain, excitation, waveguide)
        return E, A

    def getFrequency(self):
        return self.solver.getFrequency()

    def getdelx(self):
        return self.solver.getdelx()

    def getksquared(self):
        return self.solver.getksquared()

    def setFrequency(self, nu):
        self.solver.setFrequency(nu)

    def func(self, domain, Eobs, Pj_obs, excitation, waveguide):
        func = self.solver.func(domain, Eobs, Pj_obs, excitation, waveguide)
        return func
    def grad(self, domain, Eobs, Pj_obs, excitation, waveguide):
        grad = self.solver.grad(domain, Eobs, Pj_obs, excitation, waveguide)
        return grad