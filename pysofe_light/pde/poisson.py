"""
Provides the data structure representing the Poisson equation.
"""

# IMPORTS
import numpy as np
from scipy.sparse import linalg as sparse_linalg

from .base import PDE

class Poisson(PDE):
    """
    Partial differential equation for the linear Poisson equation
    
    .. math::
       - a \Delta u = f
    
    Parameters
    ----------
    
    fe_space : pysofe.spaces.FESpace
        A finite element space for which to solve the pde
    
    a : callable
        The coefficient function
    
    f : callable
        The right hand site
    
    bc : tuple
        A sequence of pysofe.base.BoundaryContition's
    """

    def __init__(self, fe_space, a=1., f=0., bc=None):
        PDE.__init__(self, fe_space, bc)

    def assemble(self):
        # compute stiffness matrix
        A = self.fe_space.assemble_laplace(d=self.fe_space.mesh.dimension)

        # compute load vector
        b = self.fe_space.assemble_l2_product(d=self.fe_space.mesh.dimension)

        # apply boundary conditions
        A, b = self.apply_boundary_conditions(A, b)
            
        self.stiffness = A
        self.load = b

    def solve(self):
        # assemble discrete system
        self.assemble()
        
        # transform matrix formats for faster arithmetic
        rhs = self.stiffness.tocsr()
        lhs = self.load.tocsr()
            
        # solve the discrete system
        sol = sparse_linalg.spsolve(rhs, lhs)
            
        # save the solution
        self.solution = sol

        return self.solution
