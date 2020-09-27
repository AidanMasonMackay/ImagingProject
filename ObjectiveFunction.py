import numpy as np

from ForwardProblemSolver.FPsolver import FPsolver
from ExperimentalSetup.ExperimentBuilder import ExperimentBuilder

from functions import *
from save import buildEobs

class ObjectiveFunction:
    """Creates objective function and its gradient function for a given experiment
    Attributes
    ----------
        experiment: object
            holds electric field measurements for an experiment and the relevant parameters which describe the experimental set-up
        grid: object
            grid dimension parameters and discretisation method. Note this is different to simulationGrid, which is the grid the E-field measurements were simulated on (if using simulated data)

    Methods
    ----------
        func
            objective function for a given estimate for kappa
        grad
            derivative of the objective function for a given estimate for kappa
    Notes
    ----------  """
    def __init__(self, experimental_data, grid):
        nu = experimental_data['nu']
        self.experiment = experimental_data
        self.measures = experimental_data['measures']
        self.simulationgrid, self.wgwidth, self.wgdistfromcenter = experimental_data['simulationGrid'], experimental_data['wgwidth'], experimental_data['wgdistfromcenter']
        self.num_sources, self.num_measures = experimental_data['num_sources'], experimental_data['num_measures']
        self.grid = grid

        self.solver = FPsolver(grid, nu)
        self.simulation = ExperimentBuilder(grid, self.num_measures, self.wgwidth, self.wgdistfromcenter, r_source=0)

        # Include waveguide if wgwidth has been specified
        if self.wgwidth:
            self.WithWaveguide = True
        else:
            self.WithWaveguide = False

    def func(self, kappa):
        """"Objective function summed over multiple source positions"""
        J = 0
        experiment = self.experiment
        grid = self.grid
        # Loop through observations
        for key in experiment['measures'].keys():
            angle = int(float(key))
            source = self.simulation.buildSource(angle, WithWaveguide = self.WithWaveguide)
            waveguide, excitation = source['waveguide'], source['excitation']
            Pj = self.simulation.buildProjectionMatrix()
            Eobs = buildEobs(grid, experiment, angle, Pj) # Build observations onto grid
            JTemp = self.solver.func(kappa, Eobs, Pj, excitation, waveguide)
            J += JTemp
        return J

    def grad(self, kappa):
        """"Gradient of objective function summed over multiple source positions"""
        experiment = self.experiment
        # TEMP - need to think through data type for kappa
        if type(kappa) == dict:
            ndim = kappa['eps'].shape[0]
        else:
            ndim = kappa.shape[0]
        gradient = np.zeros(ndim, dtype = complex)
        for key in experiment['measures'].keys():
            angle = int(float(key))
            source = self.simulation.buildSource(angle, WithWaveguide = self.WithWaveguide)
            waveguide, excitation = source['waveguide'], source['excitation']
            Pj = self.simulation.buildProjectionMatrix()
            Eobs = buildEobs(self.grid, experiment, angle, Pj) # Build observations onto grid
            gradientTemp = self.solver.grad(kappa, Eobs, Pj, excitation, waveguide)
            gradient += gradientTemp
        return gradient

def buildFuncAndGrad(experimental_data, grid):
    """ returns methods for calculating objective function and it's gradient """
    oF = ObjectiveFunction(experimental_data, grid)
    func = oF.func
    grad = oF.grad
    return func, grad