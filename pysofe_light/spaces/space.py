"""
Provides the data structures that represents a finite element space.
"""

# IMPORTS
import numpy as np

from scipy import sparse

from .. import quadrature
from .manager import DOFManager

# DEBUGGING
from IPython import embed as IPS

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
        Returns a boolean array specifying the degrees of freedom 
        associated with the mesh entities of topological dimension `d`.

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
        Evaluates the global basis functions' derivatives at given local points.

        Parameters
        ----------

        points : array_like
            The local points on the reference element

        d : int
            The derivation order

        Returns
        -------

        numpy.ndarray
            (nE x nB x nP x nD [x nD]) array containing for all elements (nE) 
            the evaluation of all basis functions first derivatives (nB) in 
            each point (nP)
        """

        if not d in (1, 2):
            raise ValueError('Invalid derivation order for global derivatives! ({})'.format(d))
        
        # evaluate inverse jacobians of the reference maps for each element
        # and given point
        # --> nE x nP x nD x nD
        inv_jac = self.mesh.ref_map.jacobian_inverse(points)
        
        # get derivatives of the local basis functions at given points
        # --> nB x nP x nD [x nD]
        local_derivatives = self.element.eval_basis(points, d=d)

        # now we decompose the matrix vector product of the inverse jacobian
        # with the local derivatives into an elementwise multiplication and
        # summation along an axis
        # therefore we have to expand some dimensions for the multiplication part
        if d == 1:
            derivatives = (inv_jac[:,None,:,:,:] * local_derivatives[None,:,:,:,None]).sum(axis=-2)    # nE x nB x nP x nD
        elif d == 2:
            derivatives = (inv_jac.swapaxes(-2,-1)[:,None,:,:,:,None] * local_derivatives[None,:,:,None,:,:]).sum(axis=-2)
            derivatives = (derivatives[:,:,:,:,:,None] * inv_jac[:,None,:,None,:,:]).sum(axis=-2)

        return derivatives

