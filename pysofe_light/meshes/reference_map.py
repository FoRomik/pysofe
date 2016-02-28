"""
Provides the data structure for the family of reference maps.
"""

# IMPORTS
import numpy as np

from pysofe_light.elements.simple.lagrange import P1

# DEBUGGING
from IPython import embed as IPS

class ReferenceMap(object):
    """
    Establishes a connection between the reference domain
    and the physical mesh elements.

    Parameters
    ----------

    mesh : pysofe_light.meshes.mesh.Mesh
        The mesh instance
    """

    def __init__(self, mesh):
        self._mesh = mesh

        # currently only supports sraight sided elements
        # hence linear shape element
        self._shape_elem = P1(dimension=mesh.dimension)

    def eval(self, points, d=0):
        """
        Evaluates each member of the family of reference maps
        or their derivatives at given local points.

        Provides the following forms of information:

        * Zero order information, i.e. the ordinary evaluation, is needed 
          to compute global points given their local counterparts

        * First order information is used to compute Jacobians, e.g. in 
          integral transformations

        Parameters
        ----------

        points : array_like
            The local points at which to evaluate
        
        d : int
            The derivation order

        Returns
        -------
        
        numpy.ndarray
            nE x nP x nD [x nD]
        """

        points = np.atleast_2d(points)

        # determine topological dimension of the mesh entities
        # onto which to map the given local points
        dim = np.size(points, axis=0)

        # evaluate each basis function of the shape element or
        # their derivatives (according to the order `d`) in every point
        basis = self._shape_elem.eval_basis(points, d=d) # nB x nP [x nD[x nD]]

        # get the vertex indices of the mesh entities onto which
        # the reference maps should be evaluated
        vertices = self._mesh.topology.get_entities(d=dim)

        # get the coordinates of all the entities' vertices
        coords = self._mesh.nodes.take(vertices - 1, axis=0)    # nE x nB x nD

        if d == 0:
            # basis: nB x nP
            maps = (coords[:,:,None,:] * basis[None,:,:,None]).sum(axis=1)
        elif d == 1:
            # basis: nB x nP x nD
            maps = (coords[:,:,None,:,None] * basis[None,:,:,None,:]).sum(axis=1)
        elif d == 2:
            # basis: nB x nP x nD x nD
            maps = (coords[:,:,None,:,None,None] * basis[None,:,:,None,:,:]).sum(axis=1)

        return maps

    def eval_inverse(self, points, hosts):
        """
        Evaluates the inverse maps of the reference maps corresponding to the
        given host elements at given global points.

        Parameters
        ----------

        points : array_like
            The global points for which to evaluate the inverse mappings

        hosts : array_like
            The host element for each of the global points
        """

        raise NotImplementedError()

    def jacobian_inverse(self, points):
        """
        Returns the inverse of the reference maps' jacobians 
        evaluated at given points.

        Parameters
        ----------

        points : array_like
            The local points at which to evaluate the jacobians
        """

        # evaluate 1st derivative of every reference map
        jacs = self.eval(points=points, d=1)

        if jacs.shape[2] == 1:
            jacs_inv = 1./jacs
        elif jacs.shape[2] == 2:
            jacs_inv = np.linalg.inv(jacs)
        else:
            raise NotImplementedError("Jacobian inverse not available yet!")

        return jacs_inv

    def jacobian_determinant(self, points):
        """
        Returns the determinants of the reference maps' jacobians 
        evaluated at given points.

        Parameters
        ----------

        points : array_like
            The local points at which to compute the determinants
        """

        jacs = self.eval(points=points, d=1)

        if jacs.shape[2] == 1:
            jacs_det = np.sqrt(np.power(jacs[...,0], 2).sum(axis=2))
        elif jacs.shape[2] == 2:
            jacs_det = np.linalg.det(jacs)
        else:
            raise NotImplementedError("Jacobian determinant not available yet!")

        return jacs_det
        
