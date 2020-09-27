
import numpy as np
import cv2

from functions import *


import numpy as np
import cv2

from functions import *

class FinDiff:
    def __init__(self, grid):
        self.grid = grid
        self.x, self.y, self.delx = grid['x'], grid['y'], grid['delx']  # grid dimensions in metres
        self.M, self.N, self.L = grid['M'], grid['N'], grid['L']  # Grid dimensions in cell numbers

    def buildHealthyTree(self, treeeps, treerad, backgroundeps = 1, yshift=0, xshift=0):
        """"""""" Returns kappa for set-up with waveguide and tree """""
        # Grid params
        grid = self.grid
        x, y, delx = grid['x'], grid['y'], grid['delx']  # grid dimensions in metres
        M, N, L = grid['M'], grid['N'], grid['L']  # Grid dimensions in cell numbers
        xshift, yshift = MetresToCells(xshift, delx), MetresToCells(yshift, delx)

        treerad = MetresToCells(treerad, delx) # convert to number of cells

        # Build tree by creating circle in matrix
        xvals = np.arange(- M /2, M/ 2, 1)
        yvals = np.arange(-N / 2, N / 2, 1)

        ymatrix = np.zeros([len(yvals), len(xvals)])  # Create matrix of y values at each grid point
        for ii in range(0, len(xvals)):
            ymatrix[:, ii] = yvals

        xmatrix = np.zeros([len(yvals), len(xvals)])  # Create matrix of x values at each grid point
        for ii in range(0, len(yvals)):
            xmatrix[ii, :] = xvals

        rmatrix = np.sqrt(xmatrix ** 2 + ymatrix ** 2)  # Create matrix of distance from centre at each grid point

        # Build tree
        removemask = rmatrix > treerad
        rmatrix[removemask] = 0
        keepmask = rmatrix > 0
        rmatrix[keepmask] = treeeps
        rmatrix[int(rmatrix.shape[0] / 2), int(rmatrix.shape[1] / 2)] = treeeps

        cols, rows = rmatrix.shape

        # Translate tree off from the centre
        translation_matrix = np.float32([[1, 0, xshift], [0, 1, yshift]])  # Translate
        tree = cv2.warpAffine(rmatrix, translation_matrix, (cols, rows))

        eps0mask = tree == 0
        tree[eps0mask] = backgroundeps

        eps = tree

        # Construct tree
        sigma = np.zeros(L)
        treeDict = {}
        treeDict['eps'] = eps.reshape(L)
        treeDict['sigma'] = sigma
        return treeDict

    def buildDiseaseCircle(self, eps, treerad, xshift, yshift, reldiseaserad = 1/2, diseasexshift = 0, diseaseyshift = 0):
        diseaserad = treerad*reldiseaserad
        disease = self.buildHealthyTree(eps, diseaserad, backgroundeps=0, yshift=yshift, xshift=xshift)
        return disease

    def buildDiseaseHalfMoon(self, eps, treerad, xshift, yshift):
        eps = self.buildHealthyTree(eps, treerad, backgroundeps = 0, yshift=yshift, xshift=xshift)['eps']
        eps1 = eps.reshape([self.N, self.M])
        eps1[:, :int(self.N/2+MetresToCells(xshift, self.delx))] = 0
        diseaseeps = eps1.reshape(self.L)
        disease = {}
        disease['eps'] = diseaseeps
        disease['sigma'] = np.zeros(self.L)
        return disease

    def buildDiseasedTree(self, healthyeps, diseaseeps, radius, diseasetype, xshift, yshift):
        diseasenames = ['circle', 'halfmoon']
        treeBase = self.buildHealthyTree(healthyeps, radius, yshift=yshift, xshift=xshift)
        if diseasetype == 'circle':
            disease = self.buildDiseaseCircle(diseaseeps-healthyeps, radius, xshift, yshift)
        elif diseasetype == 'halfmoon':
            disease = self.buildDiseaseHalfMoon(diseaseeps-healthyeps, radius, xshift=xshift, yshift=yshift)
        else:
            print("disease name not recognised, must use of of " + str(diseasenames)[1:-1])
        tree = addDomains(treeBase, disease)
        return tree

    def buildFreeSpace(self):
        L = self.grid['L']
        treeDict = {}
        treeDict['eps'] = np.ones(L)
        treeDict['sigma'] = np.zeros(L)
        return treeDict

    def BuildDomain(self, domain, diseasetype=None, healthyeps=None, diseaseeps=None, treerad=None, yshift = 0, xshift = 0):
        domainnames = ['freespace', 'healthyTree', 'diseasedTree']
        if domain == 'freespace':
            return self.buildFreeSpace()
        elif domain == 'healthyTree':
            return self.buildHealthyTree(healthyeps, treerad, yshift=0, xshift=0)
        elif domain == 'diseasedTree':
            return self.buildDiseasedTree(healthyeps, diseaseeps, treerad, diseasetype, xshift, yshift)
        else:
            print("domain name not recognised, must use of of " + str(domainnames)[1:-1])

    def __str__(self):
        return str(self.x) + "x" + str(self.y) + "m grid with " + str(self.delx) + "m square cells domain builder for fin diff method"

class FinEl:
    def __init__(self, grid):
        self.grid = grid

    def BuildDomain(self):
        pass

