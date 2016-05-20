"""
Provides the data structures that represent the boundary conditions
for the pdes.
"""

# IMPORTS
import numpy as np
from scipy import sparse

from ..spaces.operators import L2Projection

# DEBUGGING
from IPython import embed as IPS

class BoundaryCondition(object):
    """
    Base class for all boundary conditions.

    Parameters
    ----------

    fe_space : pysofe.spaces.space.FESpace
        The considered function space

    domain : callable
        A function specifying the boundary part where
        the condition should hold
    """

    def __init__(self, fe_space, domain):
        self.fe_space = fe_space
        self.domain = domain

    def apply(self, A=None, b=None):
        """
        Applies the boundary condition to the stiffness matrix `A`
        and/or load vector `b`.
        """
        raise NotImplementedError()

class DirichletBC(BoundaryCondition):
    """
    This class represents Dirichlet boundary conditions of the form
    
    .. math::
    
       u(x,t) = u_{D}|_{\\Gamma_{D}}, \\Gamma_{D}\\subseteq\\partial\\Omega
    
    Parameters
    ----------
    
    fe_space : pysofe.spaces.space.FESpace
        The considered function space

    domain : callable
        A function specifying the boundary part where
        the condition should hold
    
    g : callable
        A function specifying the values at the boundary
    """

    def __init__(self, fe_space, domain, g=0):
        BoundaryCondition.__init__(self, fe_space, domain)

        self.g = g

    def apply(self, A=None, b=None):
        # first we need to know the boundary facets
        location_mask = self.fe_space.mesh.boundary(fnc=self.domain)

        # then we need the dofs that are associated with those entities
        dim = self.fe_space.mesh.dimension - 1
        dir_dof_mask = self.fe_space.extract_dofs(d=dim, mask=location_mask)
        dir_dof_ind = dir_dof_mask.nonzero()[0]

        # replace every row of a dirichlet dof
        # with a row that has the single value 1
        # in the corresponding column
        if not sparse.isspmatrix_lil(A):
            raise TypeError('Wrong sparse matrix format ({})'.format(A.format))
        
        dir_dof_rows = dir_dof_ind[:,None].tolist()
        dir_dof_data = [[1.]] * dir_dof_ind.size
        
        A.rows[dir_dof_ind] = dir_dof_rows
        A.data[dir_dof_ind] = dir_dof_data

        # set the corresponding load vector entries as the
        # l2 projection of the boundary function
        l2_proj = L2Projection.project(fnc=self.g,
                                       fe_space=self.fe_space,
                                       codim=1,
                                       mask=None)

        b[dir_dof_ind] = l2_proj[dir_dof_ind, None]

        return A, b

class NeumannBC(BoundaryCondition):
    r"""
    Represents Neumann boundary conditions of the form

    .. math::
       \frac{\partial u}{\partial n} = h|_{\Gamma_{N}\subseteq\partial\Omega}

    where :math:`n` denotes the outer unit normal on :math:`\Gamma_{N}`.

    Parameters
    ----------

    fe_space : pysofe.spaces.space.FESpace
        The considered function space

    domain : callable
        A function specifying the boundary part where
        the condition should hold
    
    h : callable
        A function specifying the values at the boundary
    """

    def __init__(self, fe_space, domain, h=0):
        BoundaryCondition.__init__(self, fe_space, domain)

        self.h = h

    def apply(self, A=None, b=None):
        if b is not None:
            # first we need to know the boundary facets
            location_mask = self.fe_space.mesh.boundary(fnc=self.domain)

            # then we need the dofs that are associated with those entities
            dim = self.fe_space.mesh.dimension - 1
            neu_dof_mask = self.fe_space.extract_dofs(d=dim, mask=location_mask)
            neu_dof_ind = neu_dof_mask.nonzero()[0]
            
            # add the l2 projection of the boundary function
            # to the corresponding load vector entries
            l2_proj = L2Projection.project(fnc=self.h,
                                           fe_space=self.fe_space,
                                           codim=1,
                                           mask=None)
            
            b[neu_dof_ind] += l2_proj[neu_dof_ind, None]

        return A, b

