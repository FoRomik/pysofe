"""
Provides the base class for all pde classes.
"""

# IMPORTS
import numpy as np

class PDE(object):
    """
    Base class for all partial differential equations.

    Parameters
    ----------

    fe_space : pysofe.spaces.FESpace
        The functions space in which to search the solution

    bc : pysofe.pde.BoundaryCondition
        The boundary condition the solution should comply with
    """

    def __init__(self, fe_space, bc):
        self.fe_space = fe_space
        self.bc = bc
        
        # init stiffness and load
        self.stiffness = None
        self.load = None
        
        # we don't have a solution yet
        self.solution = None

        def assemble(self):
        '''
        Assembles all discrete operators.
        '''
        raise NotImplementedError()

    def apply_boundary_conditions(self, A, b):
        A, b = pbc.apply(A=A, b=b)
        return A, b
        
    def solve(self):
        '''
        Solves the pde.

        Returns
        -------

        array_like
            Computed values for the degrees of freedom
        '''
        raise NotImplementedError()

