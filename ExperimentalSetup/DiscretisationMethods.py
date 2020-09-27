
import numpy as np
from functions import *

wgsig = 1E7 # sigma value for waveguides

class FinDiff:
    """Build components for the experimental set-up using the finite differences method
    Attributes
    ----------
        grid: object
            represent finite differences grid dimensions
        wgwidth: float
            waveguide width in metres
        wgdistfromcenter: float
            distance from entrance of waveguide to the center of the grid in metres
        r_source: float
            distance from point source to center of grid for the case where a waveguide is NOT used

    Methods
    ----------
    BuildSource
        returns source object including waveguide and excitation
    BuildProjectionMatrix
        returns projection matrix which projects onto measurement locations
    Notes
    ----------  """
    def __init__(self, grid, wgwidth, wgdistfromcenter, r_source=0):
        self.grid = grid
        self.wgwidth, self.wgdistfromcenter, self.r_source = wgwidth, wgdistfromcenter, r_source

    def build_WgTop(self):
        delx = self.grid['delx']  # grid dimensions in metres
        M, N, L, P = self.grid['M'], self.grid['N'], self.grid['L'], self.grid['P']  # Grid dimensions in cell numbers

        wgwidth, wgdistfromcenter = MetresToCells(self.wgwidth, delx), MetresToCells(self.wgdistfromcenter, delx)
        wglen = int(M/2 - wgdistfromcenter)

        sigma = np.zeros(L)
        for i in range(0, wglen):
            sigma[int(L - (N / 2 - wgwidth / 2 + N * i))] = wgsig
            sigma[int(L - (N / 2 + wgwidth / 2 + N * i))] = wgsig

        eps = np.zeros(L)
        WgTop = {}
        WgTop['sigma'] = sigma
        WgTop['eps'] = eps
        return WgTop

    def build_WgBottom(self):
        delx = self.grid['delx']  # grid dimensions in metres
        M, N, L, P = self.grid['M'], self.grid['N'], self.grid['L'], self.grid['P']  # Grid dimensions in cell numbers

        wgwidth, wgdistfromcenter = MetresToCells(self.wgwidth, delx), MetresToCells(self.wgdistfromcenter, delx)
        wglen = int(M/2 - wgdistfromcenter)

        sigma = np.zeros(L)

        for i in range(0, wglen):
            sigma[int(N / 2 - wgwidth / 2 + N * i)] = wgsig
            sigma[int(N / 2 + wgwidth / 2 + N * i)] = wgsig

        eps = np.zeros(L)
        WgBottom = {}
        WgBottom['sigma'] = sigma
        WgBottom['eps'] = eps
        return WgBottom

    def build_WgRight(self):
        delx = self.grid['delx']  # grid dimensions in metres
        M, N, L, P = self.grid['M'], self.grid['N'], self.grid['L'], self.grid['P']  # Grid dimensions in cell numbers

        wgwidth, wgdistfromcenter = MetresToCells(self.wgwidth, delx), MetresToCells(self.wgdistfromcenter, delx)
        wglen = int(N/2) - wgdistfromcenter

        sigma = np.zeros(L)
        wgarray = np.ones(int(wglen)) * wgsig

        lower = int(M * N / 2 + N - int(wgwidth / 2) * N)  # Position of top of waveguide
        upper = int(M * N / 2 + N + int(wgwidth / 2) * N)  # Position of bottom of waveguide
        sigma[lower - len(wgarray): lower] = wgarray  # Place top of waveguide into sigma
        sigma[upper - + len(wgarray): upper] = wgarray  # Place bottom of waveguide into sigma

        eps = np.zeros(L)
        WgRight = {}
        WgRight['sigma'] = sigma
        WgRight['eps'] = eps
        return WgRight

    def build_WgLeft(self):
        delx = self.grid['delx']  # grid dimensions in metres
        M, N, L, P = self.grid['M'], self.grid['N'], self.grid['L'], self.grid['P']  # Grid dimensions in cell numbers

        wgwidth, wgdistfromcenter = MetresToCells(self.wgwidth, delx), MetresToCells(self.wgdistfromcenter, delx)
        wglen = int(N/2) - wgdistfromcenter

        # wg on left
        sigma = np.zeros(L)
        wgarray = np.ones(int(wglen)) * wgsig

        lower = int(M * N / 2 - int(wgwidth / 2) * N)  # Position of top of waveguide
        upper = int(M * N / 2 + int(wgwidth / 2) * N)  # Position of bottom of waveguide

        sigma[lower: lower + len(wgarray)] = wgarray  # Place top of waveguide into sigma
        sigma[upper: upper + len(wgarray)] = wgarray  # Place bottom of waveguide into sigma

        eps = np.zeros(L)
        WgLeft = {}
        WgLeft['sigma'] = sigma
        WgLeft['eps'] = eps
        return WgLeft

    def buildExcitation(self, angle, ForWaveguide = True):
        delx = self.grid['delx']  # grid dimensions in metres
        M, N, L, P = self.grid['M'], self.grid['N'], self.grid['L'], self.grid['P']  # Grid dimensions in cell numbers

        theta = (angle-90)*np.pi/180
        if ForWaveguide == True:
            r_source = M / 2 - P - 2  # Distance of source from centre of grid for waveguide. Source is just outside of the pml
        else:
            r_source = MetresToCells(self.r_source, delx) # distance from wource to center of grid
        y_source = r_source * np.sin(theta)
        x_source = r_source * np.cos(theta)

        x_source_pos, y_source_pos = int(M / 2 + x_source), int(N / 2 + y_source)

        N_source = x_source_pos * N + y_source_pos

        excitation = np.zeros(L)
        excitation[N_source] = 1
        return excitation

    def buildWaveGuide(self, angle):
        """Returns waveguide sigma and eps arrays and corresponding excitation vector"""
        assert (angle == 0 or angle == 90 or angle == 180 or angle == 270)
        excitation = self.buildExcitation(angle, ForWaveguide = True)
        if angle == 0:
            waveguide = self.build_WgLeft()
        if angle == 90:
            waveguide = self.build_WgTop()
        if angle == 180:
            waveguide = self.build_WgRight()
        if angle == 270:
            waveguide = self.build_WgBottom()
        return waveguide, excitation


    def buildSource(self, angle, WithWaveguide = False):
        L = self.grid['L']
        if WithWaveguide:
            waveguide, excitation = self.buildWaveGuide(angle)
        else:
            eps = np.zeros(L)
            sigma = np.zeros(L)
            waveguide = {}
            waveguide['sigma'] = sigma
            waveguide['eps'] = eps
            excitation = self.buildExcitation(angle, ForWaveguide = False)
        return {'waveguide': waveguide, 'excitation': excitation}

    def buildProjectionMatrix(self, num_measures, start_angle=0, measDistance=None):
        """Builds matrix which projects onto measurement locations"""
        delx = self.grid['delx']  # grid dimensions in metres
        M, N, L, P = self.grid['M'], self.grid['N'], self.grid['L'], self.grid['P']  # Grid dimensions in cell numbers

        if not measDistance:
            measDistance = self.wgdistfromcenter
        measDistance = MetresToCells(measDistance, delx)

        max_angle = 360 - 360 / (num_measures)
        angles = np.linspace(0, max_angle, num_measures)  # All angles in degrees
        angles = np.round(angles, 0)

        # Initialise array of source positions
        N_measures = np.zeros(int(num_measures))

        for i in range(0, len(angles)):  # Find x,y position of each source
            angle = angles[i]
            theta = np.radians(angle) - np.pi / 2 + np.radians(start_angle)

            # Ignoring measurement positions for now
            y_measure = measDistance * np.sin(theta)
            x_measure = measDistance * np.cos(theta)

            x_measure_pos, y_measure_pos = int(M / 2 + x_measure), int(N / 2 + y_measure)

            N_measure = x_measure_pos * N + y_measure_pos
            N_measures[i] = N_measure  # Vector of source ids
            N_measures = N_measures.astype(int)  # Convert to integers for indexing

        Pj_obs = np.zeros(L)
        for i in N_measures:
            Pj_obs[i] = 1

        return Pj_obs

    def __str__(self):
        return str(self.x) + "x" + str(self.y) + "m grid with " + str(self.delx) + "m square cells experiment component builder for fin diff method"

class FinEl:
    def __init__(self):
        pass
    def buildSetup(self):
        pass
