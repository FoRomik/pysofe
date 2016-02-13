"""
Provides classes for some simple Lagrange type finite elements.
"""

# IMPORTS
import numpy as np

from ..base import Element

class P1(Element):
    """
    Linear simplical Lagrange element.

    Parameters
    ----------

    dimension : int
        The spatial dimension of the element
    """

    is_nodal = True

    def __init__(self, dimension):
        order = 1
        n_basis = (2, 3, 4)[:dimension]
        n_verts = (2, 3, 4)[:dimension]
        
        Element.__init__(self, dimension, order, n_basis, n_verts)

        self._dof_tuple = (1, 0, 0)

    def _eval_d0basis(self, points):
        # determine number of points and their dimension
        nD, nP = points.shape

        # get number of basis functions associated with
        # the entities of this dimension
        nB = self.n_basis[nD - 1]

        # evaluate the basis functions
        basis = np.zeros((nB, nP))

        basis[0] = 1. - points.sum(axis=0)
        for i in xrange(nD):
            basis[i+1] = points[i]

        return basis

    def _eval_d1basis(self, points):
        # determine number of points and their dimension
        nD, nP = points.shape

        # get number of basis functions associated with
        # the entities of this dimension
        nB = self.n_basis[nD - 1]

        # evaluate the basis functions
        basis = np.zeros((nB, nP, nD))

        basis[0] = -1.
        for i in xrange(nD):
            basis[i+1,:,i] = 1.

        return basis

    def _eval_d2basis(self, points):
        # determine number of points and their dimension
        nD, nP = points.shape

        # get number of basis functions associated with
        # the entities of this dimension
        nB = self.n_basis[nD - 1]

        # evaluate the basis functions
        basis = np.zeros((nB, nP, nD, nD))

        return basis
