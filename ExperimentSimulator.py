
from ForwardProblemSolver.FPsolver import FPsolver
from ExperimentalSetup.ExperimentBuilder import ExperimentBuilder
from DomainBuilder.DomainBuilder import DomainBuilder
from functions import *

class simulator:
    """Simulates electric field measurements for chosen experimental set-up
    Attributes
    ----------
        simulationgrid: object
            grid dimension parameters and discretisation method
        wgwidth: float
            width of waveguide in metres. If wgwidth==None, source is considered a point source with no waveguide
        wgdistfromcenter:
            distance from center of domain to waveguide entrance in metres
        num_sources: int
            number of sources used in experiment
        num_measures: int
            number of measurement locations for each sources
        r_source: float
            only used if the experiment contains NO waveguide, so point sources are used instead. Represents distance from center of domain to point source locations
        fixMeasureLocations: bool
            if True, measure locations are the same for every source.
            if False, the measurement locations for each source will include a measurement at the entrance of the waveguide. This is used for experiments where the signal is sent and revieved from the same location
    Methods
    -------
    simulateData: returns object including simulated E-field measurements and experiment attributes with optional multiplicative noise
    setDomain: sets domain for the simulated experiment. For example the domain might be a diseased tree surrounded by air
    setFrequency: sets the frequency for the forward problem solver
    Notes
    -----
    """

    def __init__(self, simulationgrid, wgwidth, wgdistfromcenter, num_sources, num_measures, nu, r_source=0, fixMeasureLocations=False):
        self.solver = FPsolver(simulationgrid, nu)
        self.simulation = ExperimentBuilder(simulationgrid, num_measures, wgwidth, wgdistfromcenter, r_source)
        self.domainBuilder = DomainBuilder(simulationgrid)

        self.setSourceAngles(num_sources)
        self.simulationgrid, self.wgwidth, self.wgdistfromcenter = simulationgrid, wgwidth, wgdistfromcenter
        self.num_sources, self.num_measures = num_sources, num_measures

        # Include waveguide if wgwidth has been specified
        if wgwidth:
            self.WithWaveguide = True
        else:
            self.WithWaveguide = False

    def setDomain(self, domain, diseasetype=None, healthyeps=None, diseaseeps=None, treerad=None, yshift = 0, xshift = 0):
        """ build domain to solve forward problem on. For example, a diseased tree """
        self.domain = self.domainBuilder.BuildDomain(domain, diseasetype, healthyeps, diseaseeps, treerad, yshift, xshift)

    def setFrequency(self, nu):
        self.solver.setFrequency(nu)

    def setSourceAngles(self, num_sources):
        max_angle = 360 - 360 / (num_sources)
        angles = np.linspace(0, max_angle, num_sources)  # All angles in degrees
        self.source_angles = np.round(angles, 0)

    def simulateData(self, noise_factor=0):
        """ Simulates electric field measurements for a given experimental set-up """
        domain = self.domain
        angles = self.source_angles
        measures = {} # Object for saving measurements at each source position
        for angle in angles:
            Pj = self.simulation.buildProjectionMatrix()
            source = self.simulation.buildSource(angle, WithWaveguide = self.WithWaveguide)
            waveguide, excitation = source['waveguide'], source['excitation']
            E, _ = self.solver.solve(domain, excitation, waveguide)
            E += E * noise_factor
            Measures = E[Pj == 1]  # Project electric field onto measurement locations
            measures[str(angle)] = {'realPart': (np.real(Measures)).tolist(),
                                     'imagPart': (np.imag(Measures)).tolist()}

        # Retrieve all variables so object contains a record of the simulation parameters
        num_measures = self.simulation.num_measures
        num_sources = self.num_sources
        wgwidth = self.wgwidth
        wgdistfromcenter = self.wgdistfromcenter
        grid = self.simulationgrid
        nu = self.solver.getFrequency()

        data = {'measures': measures, 'num_measures': num_measures,
                'num_sources': num_sources, 'wgwidth': wgwidth, 'wgdistfromcenter': wgdistfromcenter, 'nu': nu,
                'kappaTrueProfile': {'eps': domain['eps'].tolist(),
                                     'sigma': domain['sigma'].tolist()}, 'simulationGrid': grid,
                'noise_factor': noise_factor}
        return data