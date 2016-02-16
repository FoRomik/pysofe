"""
Provides the base class for all quadrature rules.
"""

import numpy as np

class QuadRule(object):
    """
    Provides an abstract base class for all quadrature rules.

    Parameters
    ----------

    order : int
        The polynomial order up to which the quadrature should be exact
    """

    def __init__(self, order):
        self.order = order
        self._points = None
        self._weights = None

        self.set_data()

    def set_data(self):
        """
        Set the quadrature points and weights.
        """
        raise NotImplementedError()

    @property
    def points(self):
        return self._points.copy()

    @property
    def weights(self):
        return self._weights.copy()

