"""
Provides the data structures for common boundary conditions.
"""

# IMPORTS
import numpy as np
from scipy import sparse


class BoundaryCondition(object):
    """
    Base class for all boundary conditions.

    Parameters
    ----------

    fe_space : pysofe.spaces.FESpace
        ...

    domain : callable
        A function specifying the domain where the condition should hold
    """

    def __init__(self, fe_space, domain):
        self.fe_space = fe_space
        self.domain = domain

    def apply(self, A=None, b=None):
        """
        Imposes the boundary condition by altering the stiffness matrix `A`
        and/or load vector `b` of the pde.

        A : scipy.sparse.spmatrix
            The sparse stiffness matrix

        b : array_like
            The load vector
        """

        raise NotImplementedError()

class DirichletBC(BoundaryCondition):
    """
    This class represents Dirichlet boundary conditions.
    
    .. math::
    
       u(x,t) = u_{D}|_{\\Gamma_{D}}, \\Gamma_{D}\\subseteq\\partial\\Omega
    
    Parameters
    ----------
    
    fe_space : pysofe.spaces.FESpace
        ...

    domain : callable
        A function or string specifying the part of the 
        pde domain for which the boundary conditions should apply
    
    ud : callable
        A function specifying the values at the boundary    
    """
    
    def __init__(self, fe_space, domain, ud=0.0):
        BoundaryCondition.__init__(self, fe_space, domain)
        
        # set boundary function
        if callable(ud):
            self.ud = ud
        else:
            # if boundary function is given as a constant
            self.ud = lambda x: ud + 0.0 * x[0]
    
    def apply(self, A=None, b=None):
        '''
        Applies the Dirichlet boundary conditions.

        See :meth:`pysofe.base.condition.BoundaryCondition.apply`
        '''

        location = self.fe_space.mesh.get_boundary(fnc=self.domain)

        dir_dof = self.fe_space.extract_dofs(codim=1, mask=location)
        dir_dof_ind = dir_dof.nonzero()[0]

        if A is not None:
            # replace every row of a dirichlet dof
            # with a row that has the single value 1
            # in the corresponding column
            if not sparse.isspmatrix_lil(A):
                raise TypeError('Wrong sparse matrix format ({})'.format(A.format))

            dir_dof_rows = dir_dof_ind[:,None].tolist()
            dir_dof_data = [[1.]] * dir_dof_ind.size

            A.rows[dir_dof_ind] = dir_dof_rows
            A.data[dir_dof_ind] = dir_dof_data
                
        if b is not None:
            b[dir_dof_ind] = self.fe_space.eval_l2_projection(f=self.ud, d=self.fe_space.mesh.dimension-1, mask=None)[dir_dof_ind, new]

        return A, b
