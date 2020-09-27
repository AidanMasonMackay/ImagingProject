
from ExperimentalSetup.DiscretisationMethods import FinDiff, FinEl
from functions import *

wgsig = 1E7 # sigma value for waveguides
class ExperimentBuilder:
    """ Represents the experimental set-up.
    Attributes
    ----------
    Configurations : object
        Represents all microwave sources in experiment. Each configuration has four attributes
            excitation: 1darray representing point source
            waveguide: ndarray representing waveguide
            Pj_obs: ndarray representing projection matrix onto measurement locations
            Eobs: ndarray representing e-field measurements for this source position

    Methods
    -------
    BuildSource: Constructs source object
    BuildProjectionMatrix: Constructs matrix which projects onto measurement locations
    Notes
    -----
    """
    def __init__(self, grid, num_measures, wgwidth=None, wgdistfromcenter=None, measDistance=0, r_source = 0):
        discretisationMethod = grid['discretisationMethod']
        if discretisationMethod == 'FinDiff':
            self.method = FinDiff(grid, wgwidth, wgdistfromcenter, r_source) # Create an instance of discretisation method
        if discretisationMethod == 'FinEl':
            self.method = FinEl(grid, wgwidth, wgdistfromcenter, r_source) # Create an instance of discretisation method
        self.measDistance = measDistance
        self.num_measures = num_measures

    def buildSource(self, angle, WithWaveguide = False):
        """Returns waveguide sigma and eps arrays and corresponding excitation vector"""
        source = self.method.buildSource(angle, WithWaveguide = WithWaveguide)
        return source

    def buildProjectionMatrix(self, start_angle=0, measDistance=None):
        """Builds matrix which projects onto measurement locations"""
        Pj_obs = self.method.buildProjectionMatrix(self.num_measures, start_angle, self.measDistance)
        return Pj_obs

    def get_grid(self):
        return self.method.get_grid()

    def get_num_measures(self):
        return self.method.get_num_measures()

    def get_measDistance(self):
        return self.method.get_measDistance()

    def get_num_sources(self):
        return self.method.get_num_sources()

    def get_wgwidth(self):
        return self.method.get_wgwidth()

    def get_wgdistfromcenter(self):
        return self.method.get_wgdistfromcenter()

    def get_angles(self):
        return self.method.get_angles()