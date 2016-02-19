"""
Provides the data structures that represents a finite element space.
"""

# IMPORTS
import numpy as np

from scipy import sparse

from .. import quadrature
from .manager import DOFManager

class FESpace(object):
    """
    Base class for all finite element spaces.

    Connects the mesh and reference element via a
    degrees of freedom manager.

    Parameters
    ----------

    mesh : pysofe.meshes.Mesh
        The mesh used for approximating the pde domain

    element : pysofe.elements.base.Element
        The reference element
    """

    def __init__(self, mesh, element):
        # consistency check
        if not mesh.dimension == element.dimension:
            msg = "Dimension mismatch between mesh and reference element! ({}/{})"
            raise ValueError(msg.format(mesh.dimension, element.dimension))
        elif not mesh.ref_map.shape_elem.n_verts == element.n_verts:
            raise ValueError("Incompatible shapes between mesh and reference element!")

        self.mesh = mesh
        self.element = element

        # get the degrees of freedom manager
        self.dof_manager = DOFManager(mesh, element)

        # get quadrature rule
        self.quad_rules = self._get_quadrature_rules()

    @property
    def n_dof(self):
        """
        The total number of degrees of freedom
        """
        #return self._get_n_dof()
        return self.dof_manager.get_n_dof()

    def _get_dof_map(self, d, mask):
        """
        Returns the degrees of freedom mapping that connects the global mesh
        entities of topological dimension `d` to the local reference element.
        
        Parameters
        ----------
        
        d : int
            The topological dimension of the entities for which to return
            the degrees of freedom mapping

        mask : array_like
            An 1d array marking certain entities of which to get the dof map
        """
        
        dof_map = self.dof_manager.get_connectivity_array(d=d)

        if mask is not None:
            mask = np.asarray(mask)
            assert mask.ndim == 1

            if mask.dtype == bool:
                dof_map = dof_map.compress(mask, axis=1)
            elif mask.dtype == int:
                dof_map = dof_map.take(mask, axis=1)
            else:
                raise TypeError("Invalid type of mask! ({})".format(mask.dtype))
        
        return dof_map

    def _get_quadrature_rules(self):
        """
        Returns the quadrature rules for the entities up to the dimension
        of the reference element.
        """

        order = 2 * self.element.order
        dim = self.element.dimension
        
        qr = [None] * (dim + 1)
        qr[0] = quadrature.GaussPoint()
        qr[1] = quadrature.GaussInterval(order)
        if dim > 1:
            qr[2] = quadrature.GaussTriangle(order)
            if dim > 2:
                qr[3] = quadrature.GaussTetrahedron(order)

        return qr

    def _get_quadrature_data(self, d):
        """
        Returns the quadrature points and weighths associated with
        the `d`-dimensional entities.

        Parameters
        ----------

        d : int
            The topological dimension of the entities for which to
            return the quadrature points and weights
        """

        points = self.quad_rules[d].points
        weights = self.quad_rules[d].weights

        return points, weights

    def extract_dofs(self, d, mask=None):
        """
        Returns the degrees of freedom associated with the mesh entities
        of topological dimension `d` as a boolean array.

        Parameters
        ----------

        d : int
            The topological dimension of the mesh entities

        mask : array_like
            An 1d array marking certain entities of which to get the dofs
        """

        # first we need the dof map
        dof_map = self._get_dof_map(d, mask)
        n_dof = self.n_dof

        # remove duplicates and dofs mapped to `0`
        dofs = np.unique(dof_map)
        dofs = np.setdiff1d(dofs, 0)

        # build array using coo sparse matrix capabilities
        col_ind = dofs - 1
        row_ind = np.zeros_like(col_ind)
        data = np.ones_like(col_ind, dtype=bool)

        dofs = sparse.coo_matrix((data, (row_ind, col_ind)), shape=(1, n_dof))

        # turn it into an 1d array
        dofs = dofs.toarray().ravel()
        
        return dofs

    def eval_global_derivatives(self, points, d=1):
        """
        Evaluates the global basis functions derivatives at given local points.

        Parameters
        ----------

        points : array_like
            The local points on the reference element

        Returns
        -------

        numpy.ndarray
            An (nE x nB x nP x nD) array containing for all elements (nE) the evaluation of all
            (nB) basis functions first derivatives in each (nP) point
        """

        if not d in (1, 2):
            raise ValueError('Invalid derivation order for global derivatives! ({})'.format(d))
        
        # evaluate inverse jacobians of the reference maps for each element
        # and given point and transpose them
        inv_jac = self.mesh.ref_map.jacobian_inverse(points) # nE x nP x nD x nD
        
        # get derivatives of the basis functions at given points
        local_derivatives = self.element.eval_basis(points, d=d) # nB x nP x nD

        # now we decompose the matrix vector product of the inverse jacobian
        # with the gradient of the basis functions into an elementwise
        # multiplication and summation along an axis
        # therefore we have to expand some dimensions for the multiplication part
        if d == 1:
            derivatives = (inv_jac[:,None,:,:,:] * local_derivatives[None,:,:,:,None]).sum(axis=-2)    # nE x nB x nP x nD
        elif d == 2:
            derivatives = (inv_jac.swapaxes(-2,-1)[:,None,:,:,:,None] * local_derivatives[None,:,:,None,:,:]).sum(axis=-2)
            derivatives = (derivatives[:,:,:,:,:,None] * inv_jac[:,None,:,None,:,:]).sum(axis=-2)

        return derivatives

    def _assemble_operator(self, entries, d, mask):
        """
        Assembles the discrete weak form of the operator.

        Parameters
        ----------

        codim : int
            The codimension of the mesh entities for which to
            assemble the operator

        entity_mask : array_like
            Boolean array that specifies for which entities to 
            assemble the operator

        Returns
        -------

        scipy.sparse.lil_matrix
        """

        # compute indices of the entries w.r.t. the dof map
        dof_map = self.fe_space._get_dof_map(d)
        n_dof = self.fe_space.n_dof

        if mask is not None:
            assert mask.ndim == 1
            assert mask.dtype == bool

            entries = entries.compress(condition=mask, axis=0)
            dof_map = dof_map.compress(condition=mask, axis=1)

        if np.size(entries, axis=2) == 1:
            row_ind = dof_map.ravel(order='F')
            col_ind = np.ones(np.size(dof_map))
            shape = (n_dof, 1)
        else:
            nB = np.size(entries, axis=1)
            row_ind = np.tile(dof_map, reps=(nB,1)).ravel(order='F')
            col_ind = np.repeat(dof_map, repeats=nB, axis=0).ravel(order='F')
            shape = (n_dof, n_dof)

        # make entries one dimensional
        entries = entries.ravel(order='C') # !!! 
            
        # assemble
        M = sparse.coo_matrix((entries, (np.abs(row_ind)-1, np.abs(col_ind)-1)), shape=shape)
        
        return M.tolil()
    
    def assemble_mass_matrix(self, c, d, mask):
        # get quadrature points and weights for the integration
        qpoints, qweights = self._get_quadrature_data(d)    # nD x nP, nP
        
        # get jacobian determinant for the integral transformation
        jac_det = self.mesh.ref_map.jacobian_determinant(points=qpoints)    # nE x nP

        # evaluate factor
        if callable(c):
            C = self.mesh.eval_function(fnc=c, points=qpoints)              # nE x nP
            C = C[:,new,new,:]                                              # nE x  1 x  1 x nP
        else:
            assert isinstance(c, (int, float))
            C = c
        
        # compute entries of the operator matrix
        basis = self.element.eval_basis(points=qpoints, d=0)
        values = basis[new,new,:,:] * basis[new,:,new,:]                    # nE x nB x nB x nP
        
        jac_det = jac_det[:,new,new,:]                                      # nE x  1 x  1 x nP
        qweights = qweights[new,new,new,:]                                  #  1 x  1 x  1 x nP

        try:
            entries = (C * values * jac_det * qweights).sum(axis=-1)            # nE x nB x nB
        except MemoryError as err:
            if isinstance(c, (int, float)) and self.mesh.shape_elem.order == 1:
                # if mesh facets are straight sided the jacobian determinant
                # is constant for all elements/points
                # if the function is constant too we hopefully can avoid the memory error
                entries = (C * values * jac_det[0,0,0,0] * qweights).sum(axis=-1)
                entries = np.tile(entries, reps=(jac_det.shape[0],1,1))
            else:
                raise err

        return self._assemble_operator(entries, d, mask)

    def assemble_l2_product(self, f, d, mask):
        # get quadrature points and weights for the integration
        qpoints, qweights = self._get_quadrature_data(d)    # nD x nP, nP
        
        # get jacobian determinant for the integral transformation
        jac_det = self.mesh.ref_map.jacobian_determinant(points=qpoints)    # nE x nP

        # evaluate factor
        if callable(f):
            F = self.mesh.eval_function(fnc=f, points=qpoints)              # nE x nP
            F = F[:,new,new,:]                                              # nE x  1 x  1 x nP
        else:
            assert isinstance(f, (int, float))
            F = f
        
        # compute entries of the operator matrix
        basis = self.element.eval_basis(points=qpoints, d=0)                # nB x nP
        values = basis[new,:,new,:]                                         #  1 x nB x  1 x nP
        
        jac_det = jac_det[:,new,new,:]                                      # nE x  1 x  1 x nP
        qweights = qweights[new,new,new,:]                                  #  1 x  1 x  1 x nP

        entries = (F * values * jac_det * qweights).sum(axis=-1)            # nE x nB x  1

        return self._assemble_operator(entries, d, mask)

    def assemble_laplace(self, a, d, mask):
        # get quadrature points and weights for the integration
        qpoints, qweights = self._get_quadrature_data(d)    # nD x nP, nP
        
        # get jacobian determinant for the integral transformation
        jac_det = self.mesh.ref_map.jacobian_determinant(points=qpoints)    # nE x nP

        # evaluate factor
        A = self.mesh.eval_function(fnc=a, points=qpoints)       # nE x nP[x nD[x nD]]
                
        # compute entries of the operator matrix
        dbasis_global = self.eval_global_derivatives(points=qpoints) # nE x nB x nP x nD
        nE, nB, nP, nD = dbasis_global.shape

        db_g = dbasis_global                                                  # nE x nB x nP x nD
        
        if (A.ndim - 2) == 0:
            # assuming A is scalar
            A = A[:,new,new,:]                                                # nE x  1 x  1 x nP
            values = A * (db_g[:,new,:,:,:] * db_g[:,:,new,:,:]).sum(axis=-1) # nE x nB x nB x nP
        elif (A.ndim - 2) == 2:
            # assuming A is a matrix
            A = A[:,None,:,:,:]                                               # nE x  1 x nP x nD x nD
            Adb_g = (A * db_g[:,:,:,None,:]).sum(axis=-1)                     # nE x nB x nP x nD
            values = (Adb_g[:,new,:,:,:] * db_g[:,:,new,:,:]).sum(axis=-1)    # nE x nB x nB x nP

        jac_det = jac_det[:,new,new,:]                               # nE x  1 x  1 x nP
        qweights = qweights[new,new,new,:]                           #  1 x  1 x  1 x nP

        entries = (values * jac_det * qweights).sum(axis=-1)     # nE x nB x nB

        return self._assemble_operator(entries, d, mask)

    def eval_l2_projection(self, f, d, mask=None):
        # assemble L2 product
        F = self.assemble_l2_product(f, d, mask)

        #  assmeble mass matrix
        M = self.assemble_mass_matrix(c=1, d=d, mask=mask)

        U = np.zeros(np.size(F))
        dof_map = self._get_dof_map(d)
        I = np.setdiff1d(np.unique(np.abs(dof_map)), 0) - 1

        M = M.tocsr()
        F = F.tocsr()

        U[I] = sparse.linalg.spsolve(M[I,:][:,I], F[I])

        return U


