import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import eigs

from functions import ConvertToKappa, ConvertKappaToDict, MetresToCells

class FinDiff:

    def __init__(self, optimisationgrid, nu, trunc_dist = 0):
        """Provides methods for AEI optimisation algorithm using the finite differences method
        Attributes
        ----------
            optimisationgrid: object
                contains optimisationgrid dimension parameters and discretisation method. Note this is different to simulationGrid, which is the grid the E-field measurements were simulated on (if using simulated data)
            trunc_dist: float
                distance (in metres) to truncate domain from the edges
        Methods
        ----------
            calcEigenFuncs
                Calculates eigenfunctions of input kappa
            setDomainMask
                sets the truncated domain at dist from edge on which to calculate the eigenfunctions
        Notes
        ----------  """
        self.optimisationgrid = optimisationgrid
        self.setDomainMask(trunc_dist)
        self.nu = nu

    def setDomainMask(self, dist):
        self.domainMask, self.Mtrunc, self.Ntrunc, self.Ltrunc = self.buildSubDomainMask(dist)

    def buildSubDomainMask(self, dist):
        """Constructs mask which projects onto the subdomain. Subdomain includes all cells more than dist metres from edge"""
        M, N, L = self.optimisationgrid['M'], self.optimisationgrid['N'], self.optimisationgrid['L']
        P, delx = self.optimisationgrid['P'], self.optimisationgrid['delx']
        Dist = int(P + MetresToCells(dist, delx))  # Choose distance from edges to crop the domain

        # N, M and L for smaller domain - this controls the domain for the eigenfuncs
        Ntrunc = N - 2 * Dist
        Mtrunc = M - 2 * Dist

        Pj_domain = np.ones(L, dtype=complex)  # Initialise projection vector
        Pj_domain[0:N * Dist] = 0
        Pj_domain[-N * Dist:] = 0
        for i in range(0, Dist):
            Pj_domain[i::N] = 0
            Pj_domain[N - 1 - i::N] = 0
        domainMask = Pj_domain == 1  # Mask to isolate truncated domain
        Ltrunc = Mtrunc*Ntrunc
        return domainMask, Mtrunc, Ntrunc, Ltrunc

    # It might be better if this function didn't deal with breaking up kappa into arrays?
    def calcEigenFunctions(self, kappa, num_eigenfunctions, sigma_val=-1e-6, which="SR", epsilon=1e-6):
        """" Computes the discrete eigenfunctions and eigenvalues of elliptical operator as per AEI method """
        M, N, L = self.optimisationgrid['M'], self.optimisationgrid['N'], self.optimisationgrid['L']
        P, delx = self.optimisationgrid['P'], self.optimisationgrid['delx']

        domainMask, Mtrunc, Ntrunc, Ltrunc = self.domainMask, self.Mtrunc, self.Ntrunc, self.Ltrunc # truncated domain
        kappa = ConvertToKappa(kappa, nu=self.nu) # Convert kappa from dict to ndarray
        kappaMasked = kappa[domainMask]

        ## Compute rho vector  ---> rho = 1/sqrt(grad(kappa)^2 + epsilon^2)

        # partial y
        diags = np.ones(Ltrunc)  # (L-1 not L because one value's lost over the boundary)
        partial_ymat = sp.sparse.spdiags([-diags, diags], [-1, 1], Ltrunc, Ltrunc).todense() / (2 * delx)
        partial_y1 = np.dot(partial_ymat, kappaMasked)
        partial_y = np.array(partial_y1)[0]  # Go from matrix to vector form

        # partial x
        partial_xmat = sp.sparse.spdiags([-diags, diags], [-Ntrunc, Ntrunc], Ltrunc, Ltrunc).todense() / (2 * delx)
        partial_x1 = np.dot(partial_xmat, kappaMasked)
        partial_x = np.array(partial_x1)[0]  # Go from matrix to vector form

        # Try work out what's happening with the line along the top - for now just remove it using Pj_remove
        Pj_remove = np.ones(Ltrunc, dtype=complex)
        Pj_remove[0:Ntrunc] = 0
        Pj_remove[-Ntrunc:] = 0
        Pj_remove[0::Ntrunc] = 0
        Pj_remove[Ntrunc - 1::Ntrunc] = 0

        # Find the length of gradient of kappa
        gradsquared = np.real(
        partial_y * partial_y.conjugate() + partial_x * partial_x.conjugate()) * Pj_remove  # real part just to change dtype, is aleady real

        rho = np.real(1 / (np.sqrt(gradsquared + epsilon ** 2)))  # This vector is real anyway, but take real part to change dtype

        # Calc midpoints for j
        rho_jplus_half = np.append((rho[:-1] + rho[1:]) / 2, rho[-1])
        rho_jminus_half = np.roll(rho_jplus_half, 1)

        # Calc midpoints for i - there's weird edge effects but they get taken out when Dirichlet conditions imposed
        rhoiplus1 = np.roll(rho, -Ntrunc)
        rhoiminus1 = np.roll(rho, +Ntrunc)

        rho_iplus_half = 0.5 * (rho + rhoiplus1)
        rho_iminus_half = 0.5 * (rho + rhoiminus1)

        middle_vec = -(rho_iminus_half + rho_jminus_half + rho_jplus_half + rho_iplus_half)

        # Build G-matrix
        G_mat = sp.sparse.diags(
            [rho_iminus_half[Ntrunc:], rho_jminus_half[1:], middle_vec, rho_jplus_half, rho_iplus_half],
            [-Ntrunc, -1, 0, 1, Ntrunc]) * (1 / delx ** 2)

        # Remove boundary values
        G_mat_dense = G_mat.tolil()

        G_mat_dense[:Ntrunc, :] = 0  # x bounds
        G_mat_dense[-Ntrunc:, :] = 0

        G_mat_dense[Ntrunc::Ntrunc] = 0  # y bounds
        G_mat_dense[Ntrunc - 1::Ntrunc] = 0

        G_mat_sparse = sp.sparse.csr_matrix(G_mat_dense)  # Matrix takes roughly 5 - 10 seconds to build for 100 x 100 grid

        #Solve for eigenfunctions in truncated domain
        values, vectors1 = eigs(G_mat_sparse, which=which, sigma=sigma_val, k=num_eigenfunctions)

        # # Pad eigenfunctions with zeros outside of truncated domain
        vectors2 = np.zeros([num_eigenfunctions, L])
        for i in range(0, num_eigenfunctions):
            eig = vectors1[:, i]
            vectors2[i][domainMask] = eig

        vectors = vectors2.T  # Final eigenvectors

        return values, vectors

    def calcLaplaceEigenfunctions(self, num_eigenfunctions, which="SR", sigma=-1e-6):
        """ Gets eigenvalues and eigenvectors of the Laplacian with Dirichlet B.Cs """
        #
        # if self.L_vals: # Used saved values if they've already been calculated
        #     return self.L_vals, self.L_vecs

        M, N, L = self.optimisationgrid['M'], self.optimisationgrid['N'], self.optimisationgrid['L']
        P, delx = self.optimisationgrid['P'], self.optimisationgrid['delx']

        # Build Laplacian
        diag = np.ones(L)
        L_mat = sp.sparse.spdiags([diag, diag, -4 * diag, diag, diag], [-N, -1, 0, 1, N], L, L) * (1 / delx ** 2)

        # Replace bounds with zeros for Dirichlet boundary conditions
        L_mat_dense = L_mat.tolil()

        L_mat_dense[:N, :] = 0  # x bounds
        L_mat_dense[-N:, :] = 0

        L_mat_dense[N::N] = 0  # y bounds
        L_mat_dense[N - 1::N] = 0

        L_mat_sparse = sp.sparse.csr_matrix(L_mat_dense)

        # Get eigenvectors
        self.L_vals, self.L_vecs = eigs(L_mat_sparse, which=which, sigma=sigma, k=num_eigenfunctions)

        return self.L_vals, self.L_vecs

class FinEl():
    pass
