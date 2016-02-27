"""
Provides the data structure for approximating the spatial domain
of the partial differential equation.
"""

# IMPORTS
import numpy as np

import refinements
from .geometry import MeshGeometry
from .topology import MeshTopology
from .reference_map import ReferenceMap

class Mesh(object):
    """
    Provides a class for general meshes.

    Basically clues together the MeshGeometry and MeshTopology
    classes and provides interfaces for mesh refinement and
    global point search.

    Parameters
    ----------

    nodes : array_like
        The coordinates of the mesh nodes

    connectivity : array_like
        The connectivity array defining the mesh cells via their vertex indices
    """

    def __init__(self, nodes, connectivity):
        # transform input arguments if neccessary
        nodes = np.atleast_2d(nodes)
        connectivity = np.asarray(connectivity, dtype=int)

        # check input arguments
        assert 1 <= nodes.shape[1] <= 3
        
        # get mesh dimension from nodes
        self._dimension = nodes.shape[1]

        # init mesh geometry and topology
        self.geometry = MeshGeometry(nodes)
        self.topology = MeshTopology(cells=connectivity, dimension=self._dimension)

        # init reference maps class
        self.ref_map = ReferenceMap(self)

    @property
    def dimension(self):
        """
        The spatial dimension fo the mesh.
        """
        return self._dimension

    @property
    def nodes(self):
        """
        The coordinates of the mesh nodes.
        """
        return self.geometry.nodes

    @property
    def cells(self):
        """
        The incident vertex indices of the mesh cells.
        """
        # the mesh cells have the same topological dimension as the mesh dimension
        return self.topology.get_entities(d=self.dimension)

    @property
    def facets(self):
        """
        The incident vertex indices of the mesh facets.
        """
        # the mesh facets have topological codimension 1
        return self.topology.get_entities(d=self.dimension - 1)

    @property
    def faces(self):
        """
        The incident vertex indices of the mesh faces.
        """
        # faces are mesh entities of topological dimension 2
        return self.topology.get_entities(d=2)

    @property
    def edges(self):
        """
        The incident vertex indices of the mesh edges.
        """
        # edges are mesh entities of topological dimension 1
        return self.topology.get_entities(d=1)

    @property
    def boundary(self):
        """
        The boundary facets of the mesh.
        """
        return self.topology.get_boundary(d=self.dimension-1)

    def get_boundary(self, fnc=None, rtype='bool'):
        """
        Determines the facets that form the mesh boundary.

        Parameters
        ----------

        fnc : callable
            Function specifying some part of the boundary for which
            to return the corresponding facets

        rtype : str
            The return type of the facets representation, either as
            a boolean array ('bool') or as an array of indices ('index')
        """

        # get a mask specifying the boundary facets
        boundary_mask = self.topology.get_boundary(d=self.dimension-1).astype(bool)

        if fnc is not None:
            assert callable(fnc)

            # to determine the facets that belong to the desired
            # part of the boundary we compute the centroids of
            # all boundary facets and pass them as arguments to
            # the given function which shall return True for all
            # those that belong to the specified part

            # to compute the centroids we need the vertex indices of
            # every facet and the corresponding mesh node coordinates
            facet_vertices = self.facets.compress(boundary_mask, axis=0)
            facet_vertex_coordinates = self.nodes.take(facet_vertices - 1, axis=0)
            centroids = facet_vertex_coordinates.mean(axis=1)

            # pass them to the given function
            try:
                part_mask = fnc(centroids)
            except:
                # given function might not be vectorized
                # so try looping over the centroids
                # --> may be slow
                ncentroids = np.size(centroids, axis=0)
                part_mask = np.empty(shape=(ncentroids,), dtype=bool)

                for i in xrange(ncentroids):
                    part_mask[i] = fnc(centroids[i,:])

            boundary_mask[boundary_mask] = np.logical_and(boundary_mask[boundary_mask], part_mask)

        if rtype == 'bool':
            return boundary_mask
        elif rtype == 'index':
            return self.facets.compress(boundary_mask, axis=0)
        else:
            raise ValueError('Invalid return type ({})'.format(rtype))


    def refine(self, method='uniform', **kwargs):
        """
        Refines the mesh using the given method.

        Parameters
        ----------

        method : str
            A string specifying the refinement method to use
        """
        refinements.refine(mesh=self, method=method, inplace=True, **kwargs)

    def eval_function(self, fnc, points):
        '''
        Evaluates given function on mesh w.r.t. given local points on reference element.

        Parameters
        ----------

        fnc : callable
            The function that should be evaluated

        points : array_like
            The local points on the reference eleemnt
        '''

        # check if given function is callable or constant
        if not callable(fnc):
            if isinstance(fnc, (int, float)):
                return fnc * np.ones((1,1))
            else:
                raise TypeError("Invalid function type for evaluation")
        
        # compute the global counterparts to the given local points
        P = self.ref_map.eval(points=points, d=0) # nE x nP x nD
        nE, nP, nD = P.shape

        # stack and transpose them so that they can be passed to the function
        P = np.vstack(P).T    # nD x nE*nP

        # evaluate the given function in each point
        try:
            F = fnc(P)    # fnc_nD x nE*nP
        except Exception as err:
            raise RuntimeError("Function evaluation returned error: {}".format(err.message))
        
        # get function image dimensions
        # by passing only one point
        f = np.atleast_1d(fnc(P.T[0][:,None]))
        
        fnc_ndim = f.ndim
        fnc_shape = f.shape

        # TODO: clarify why order C and not F???
        if fnc_ndim == 1:
            # assuming scalar
            if F.shape[0] == 1:
                # got back only one value
                # so prepare for broadcasting
                F = F[None,None]
            else:
                # assuming nE*nP values
                F = F.reshape((nE, nP), order='C')
        elif fnc_ndim == 2:
            # assuming either vector or matrix
            dim = self.dimension
            
            if F.shape[1] == 1:
                # assuming vector
                if F.shape[0] == dim:
                    F = F[None,None,:]
                else:
                    F = F.reshape((nE, nP, dim), order='C')
            elif F.shape[1] == dim:
                # assuming matrix output
                dim = self.dimension
                if F.shape[:2] == (dim, dim):
                    F = F[None,None]
                else:
                    F = F.reshape((nE, nP, dim, dim), order='C')
        elif fnc_ndim == 3:
            # assuming matrix (or vector)
            dim = self.dimension

            try:
                F = F.reshape((nE, nP, dim, dim), order='C')
            except Exception as err:
                if F.shape[-2:] == (dim, dim):
                    if F.shape[0] == 1:
                        F = F[None,:,:,:]
                    else:
                        raise err
                elif F.shape[-2:] == (dim, 1):
                    if F.shape[0] == 1:
                        F = F[None,:,:,0]
                    elif F.shape[0] == nE*nP:
                        F = F[:,:,0].reshape((nE, nP, dim))
                    else:
                        raise err
                        
        return F

