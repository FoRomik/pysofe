"""
Provides the data structures that represent the boundary conditions
for the pdes.
"""

# IMPORTS
import numpy as np
from scipy import sparse

from ..spaces.operators import L2Projection
from ..utils import match_nodes

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

class PeriodicBC(BoundaryCondition):
    """
    Represents periodic boundary conditions for n-orthotopes.

    Parameters
    ----------

    fe_space : pysofe.spaces.FESpace
        The considered function space

    master_domain : callable
        Functions specifying the master facets

    slave_domain : callable
        Functions specifying the slave facets
    """

    def __init__(self, fe_space, master_domain, slave_domain):
        BoundaryCondition.__init__(self, fe_space,
                                   domain=(master_domain, slave_domain))

        # save master and slave domains
        self._master_domain = master_domain
        self._slave_domain = slave_domain

        # cache
        self._master_dofs = None
        self._slave_dofs = None
        self._trafo_matrix = None

    def apply(self, A=None, b=None, retain_dofs=True):

        # get transformation matrix
        M = self._get_trafo_matrix(retain_dofs=retain_dofs, spformat='csr')
            
        # modify the system
        if A is not None:
            A = M.T.dot(A).dot(M) # M' * A * M

        if b is not None:
            b = M.T.dot(b)        # M' * b

        if A is not None and retain_dofs:
            A = A.tolil()

            slave_rows = np.logical_not(A.rows.astype(bool)).nonzero()[0]
            slave_data = np.ones_like(slave_rows)

            A.rows[slave_rows] = slave_rows[:,None].tolist()
            A.data[slave_rows] = slave_data[:,None].tolist()

            #A = A.tocsr()

        #IPS()

            
        # # make solution unique by setting first dof (value at lower left corner)
        # if A is not None:
        #     A = A.tolil()
        #     A[0,0] = 1.; A[0,1:] = 0.
        #     A = A.tocsr()

        # if b is not None:
        #     b = b.tolil()
        #     b[0] = 0.
        #     b = b.tocsr()
        
        return A, b

    def _get_master_slave_dofs(self):
        """
        Determines the degrees of reedom associated with the master
        and slave domains.
        """
        
        # first, we need the indices of the mesh facets
        # that form the periodic boundary
        master_facets_mask = self.fe_space.mesh.boundary(fnc=self._master_domain)
        slave_facets_mask = self.fe_space.mesh.boundary(fnc=self._slave_domain)

        # (not .nonzero()[0] + 1 since we use it as an integer mask)
        master_facets_ind = master_facets_mask.nonzero()[0]
        slave_facets_ind = slave_facets_mask.nonzero()[0]

        # the boundary nodes of the master and slave domain
        # are assumed to match, but the corresponding facets
        # might be numbered in different orders ?
        # -> not yet implemented (hope for the best)
        #I, J = self._match_facets(master_facets_ind, slave_facets_ind)
        
        # then, we use them to get the corresponding dofs
        master_dofs = self.fe_space.extract_dofs(d=1, mask=master_facets_ind,
                                                 return_indices=True)
        slave_dofs = self.fe_space.extract_dofs(d=1, mask=slave_facets_ind,
                                                return_indices=True)

        # TODO: handle possibly differently ordered facet indices
        #       -> match facets (dofs)
        
        # DEPRECATED: might only work for linear elements
        # -----------
        # # match node coordinates
        # nodes = self.fe_space.mesh.nodes
        # master_nodes_mask = [domain(nodes) for domain in self._master_domains]
        # slave_nodes_mask = [domain(nodes) for domain in self._slave_domains]

        # master_nodes =  [nodes.compress(mask, axis=0) for mask in master_nodes_mask]
        # slave_nodes = [nodes.compress(mask, axis=0) for mask in slave_nodes_mask]

        # matches = []
        # for i in xrange(2):
        #     matches.append(match_nodes(master_nodes[i], slave_nodes[i], ax=i))
        
        # master_dofs = [self.fe_space.extract_dofs(1, master_loc).toarray().ravel()
        #                for master_loc in master_locs]
        # slave_dofs = [self.fe_space.extract_dofs(1, slave_loc).toarray().ravel()
        #               for slave_loc in slave_locs]

        # master_dofs = [m.nonzero()[0] + 1 for m in master_dofs]
        # slave_dofs = [s.nonzero()[0] + 1 for s in slave_dofs]

        # for i in xrange(2):
        #     match_master, match_slave = matches[i]
        #     master_dofs[i] = master_dofs[i].take(match_master)
        #     slave_dofs[i] = slave_dofs[i].take(match_slave)

        # cache them
        self._master_dofs = master_dofs
        self._slave_dofs = slave_dofs

        return master_dofs, slave_dofs

    def _match_facets(master_facets, slave_facets):
        raise NotImplementedError()

    def _get_trafo_matrix(self, retain_dofs=True, spformat='csr'):
        if self._trafo_matrix is not None:
            return self._trafo_matrix
        
        # get master and slave dofs
        master_dofs, slave_dofs = self._get_master_slave_dofs()

        # create master-slave transformation matrix
        n_dof = self.fe_space.n_dof

        # initialise trafo matrix in lil-format
        # to facilitate altering its structure
        T = sparse.eye(n_dof, n_dof, dtype=int, format='lil')

        # apply master-slave identification
        T.rows[slave_dofs - 1] = T.rows[master_dofs - 1]

        if not retain_dofs:
            # remove zero slave columns
            master_columns = np.asarray(np.unique(T.rows).tolist(), dtype=int).flatten()
            #T = T[:,master_columns]
            T = T.tocsc()[:,master_columns]
        
        # change format
        T = T.asformat(spformat)
        
        self._trafo_matrix = T

        return T

    def recover_slave_dofs(self, dofs):
       M = self._get_trafo_matrix()

       d = M.dot(dofs)
       
       return d
