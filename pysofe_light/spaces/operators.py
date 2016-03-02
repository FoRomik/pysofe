"""
Provides data structures that represent weak forms differential operators.
"""

# IMPORTS
import numpy as np
from scipy import sparse

# DEBUGGING
from IPython import embed as IPS

class Operator(object):
    """
    Base class for all operators.

    Derived classes have to implement the method 
    :py:meth:`_compute_entries()`

    Parameters
    ----------

    fe_space : pysofe_light.spaces.space.FESpace
        A function space the operator works on
    """

    def __init__(self, fe_space):
        self.fe_space = fe_space
        
    def assemble(self, codim=0, mask=None):
        """
        Assembles the discrete weak operator.

        Parameters
        ----------

        codim : int
            The codimension of the entities for which to assemble

        mask : array_like
            Boolean 1d array marking specific entities for assembling

        Returns
        -------

        scipy.sparse.lil_matrix
        """

        # first compute the operator specific entries
        entries = self._compute_entries(codim=codim)

        raise NotImplementedError()

    def _compute_entries(self, codim=0):
        """
        Computes the entries for the discrete form of the operator.
        """
        raise NotImplementedError()
    
