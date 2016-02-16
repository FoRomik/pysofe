"""
Provides the data structures for Gauss-Legendre quadrature on simplicial domains.
"""

# IMPORTS
import numpy as np

from .base import QuadRule

class GaussQuad(QuadRule):
    """
    Base class for Gaussian quadrature rules.
    """
    
    @staticmethod
    def get_gauss_points(order):
        """
        Generates the points and weights for a 1-dimensional 
        Gauss-Legendre quadrature.

        Parameters
        ----------

        order : int
            The polynomial order up to which the quadrature should be exact
        """

        n = order + 1
        #n = int(np.ceil((order + 1)/2.))

        # get points and weights
        points, weights = np.polynomial.legendre.leggauss(n)
        
        return points, weights

class GaussPoint(QuadRule):
    """
    Gauss-Legendre quadrature rule for a single point.
    """
    
    def __init__(self):
        QuadRule.__init__(self, order=0)

    def set_data(self):
        self._points = np.empty((0,0))
        self._weights = np.array([1.])

class GaussInterval(QuadRule):
    """
    Gauss-Legendre quadrature rule for intervals.
    """

    def __init__(self, order):
        QuadRule.__init__(self, order=order)

    def set_data(self):
        points, weights = self.get_gauss_points(order=self.order)
        
        self._points = 0.5 * (points + 1.)
        self._weights = 0.5 * weights

class GaussTriangle(QuadRule):
    """
    Gauss-Legendre quadrature rule for triangles.
    """

    def __init__(self, order):
        QuadRule.__init__(self, order=order)

    def set_data(self):
        points, weights = self.get_gauss_points(self.order)
        points = 0.5 * (points + 1.)
        weights = 0.5 * weights
        
        W = np.outer(weights, weights).ravel(order='F')
        P = np.vstack([np.tile(points, (np.size(weights, 0),)), 
                       np.repeat(points, repeats=np.size(weights, 0), axis=0)])
        
        # Duffy Transform
        self._weights = (W * (1 - P[1]))
        self._points = np.vstack([P[0] * (1 - P[1]), P[1]])

class GaussTetrahedron(QuadRule):
    """
    Gauss-Legendre quadrature rule for tetrahedra.
    """

    def __init__(self, order):
        raise NotImplementedError()
