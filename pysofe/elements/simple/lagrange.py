"""
Provides classes for some simple Lagrange type finite elements.
"""

# IMPORTS
import numpy as np

from ..base import Element

class P1(Element):
    """
    Linear Lagrange basis functions on simplicial domains.

    Parameters
    ----------

    dimension : int
        The spatial dimension of the element
    """

    is_nodal = True

    def __init__(self, dimension):
        order = 1
        n_basis = (1, 2, 3, 4)[:(dimension+1)]
        n_verts = (1, 2, 3, 4)[:(dimension+1)]
        
        Element.__init__(self, dimension, order, n_basis, n_verts)

        self._dof_tuple = (1, 0, 0, 0)[:(dimension+1)]

    def _eval_d0basis(self, points):
        # determine number of points and their dimension
        nD, nP = points.shape

        # get number of basis functions associated with
        # the entities of this dimension
        nB = self.n_basis[nD]

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
        nB = self.n_basis[nD]

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
        nB = self.n_basis[nD]

        # evaluate the basis functions
        basis = np.zeros((nB, nP, nD, nD))

        return basis

class P2(Element):
    """
    Quadratic Lagrange basis functions on simplicial domains.

    Parameters
    ----------

    dimension : int
        The spatial dimension of the element
    """

    is_nodal = True
    
    def __init__(self, dimension):
        order = 2
        n_basis = (1, 3, 6, 10)[:(dimension+1)]
        n_verts = (1, 2, 3, 4)[:(dimension+1)]
        
        Element.__init__(self, dimension, order, n_basis, n_verts)

        self.dof_tuple = (1, 1, 0, 0)[:(dimension+1)]

    def _eval_d0basis(self, points):
        # determine number of points and their dimension
        nD, nP = points.shape

        # get number of basis functions associated with
        # the entities specified by given points
        nB = self.n_basis[nD]

        # now we evaluate each basis function in every point given
        basis = np.zeros((nB, nP))

        if nD == 1:
            basis[0] = 1. - points[0]
            basis[1] = points[0]
            basis[2] = 4. * points[0] * (1. - points[0])
        elif nD == 2:
            l1 = 1. - points.sum(axis=0)
            l2 = points[0]
            l3 = points[1]
            
            basis[0] = l1
            basis[1] = l2
            basis[2] = l3
            basis[3] = 4. * l1 * l2
            basis[4] = 4. * l1 * l3
            basis[5] = 4. * l2 * l3
        elif nD == 3:
            raise NotImplementedError()

        return basis

    def _eval_d1basis(self, points):
        # determine number of points and their dimension
        nD, nP = points.shape

        # get number of basis functions associated with
        # the entities specified by given points
        nB = self.n_basis[nD]

        # now we evaluate each basis function in every point given
        basis = np.zeros((nB, nP, nD))

        if nD == 1:
            basis[0,:,0] = -1.
            basis[1,:,0] = 1.
            basis[2,:,0] = 4. - 8. * points[0]
        elif nD == 2:
            x0 = points[0]
            x1 = points[1]
            
            basis[0,:,[0,1]] = -1.
            basis[1,:,0] = 1.
            basis[2,:,1] = 1.
            basis[3,:,0] = 4. * (1. - 2. * x0 - x1)
            basis[3,:,1] = -4. * x0
            basis[4,:,0] = -4. * x1
            basis[4,:,1] = 4. * (1. - x0 - 2. * x1)
            basis[5,:,0] = 4. * x1
            basis[5,:,1] = 4. * x0
        elif nD == 3:
            raise NotImplementedError()

        return basis

    def _eval_d2basis(self, points):
        # determine number of points and their dimension
        nD, nP = points.shape

        # get number of basis functions associated with
        # the entities specified by given points
        nB = self.n_basis[nD]

        # now we evaluate each basis function in every point given
        basis = np.zeros((nB, nP, nD, nD))

        if nD == 1:
            basis[2,:,0,:] = -8.
        elif nD == 2:
            basis[3,:,0,0] = -8.
            basis[3,:,0,1] = -4.
            basis[3,:,1,0] = -4.
            basis[4,:,0,1] = -4.
            basis[4,:,1,0] = -4.
            basis[4,:,1,1] = -8.
            basis[5,:,0,1] = 4.
            basis[5,:,1,0] = 4.
        elif nD == 3:
            raise NotImplementedError()

        return basis
