"""
Provides the data structure for approximating the spatial domain
of the partial differential equation.
"""

# IMPORTS
import numpy as np

class Mesh(object):
    """
    Provides a class for general meshes.

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
        assert 1 <= nodes.shape[0] <= 3
        
        # get mesh dimension from nodes
        self._dimension = nodes.shape[0]

        # init mesh geometry and topology
        self.Geometry = MeshGeometry(nodes)
        self.Topology = MeshTopology(connectivity)

        # init reference maps class
        self.RefMap = ReferenceMap(self)

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
        return self.Geometry.nodes

    @property
    def cells(self):
        """
        The incident vertex indices of the mesh cells.
        """
        # the mesh cells have the same topological dimension as the mesh dimension
        return self.Topology.get_entities(d=self.dimension)

    @property
    def facets(self):
        """
        The incident vertex indices of the mesh facets.
        """
        # the mesh facets have topological codimension 1
        return elf.Topology.get_entities(d=self.dimension - 1)

    @property
    def faces(self):
        """
        The incident vertex indices of the mesh faces.
        """
        # faces are mesh entities of topological dimension 2
        return self.Topology.get_entities(d=2)

    @property
    def edges(self):
        """
        The incident vertex indices of the mesh edges.
        """
        # edges are mesh entities of topological dimension 1
        return self.Topology.get_entities(d=1)

    def get_orientation(self, d=1):
        """
        Returns an array indicating if the vertices of the
        mesh entities with the given topological dimension
        are ordered correctly.

        Parameters
        ----------

        d : int
            The topological dimension of the entities to check
        """
        return self.Topology._get_orientation(d)

    def refine(self, method='uniform', **kwargs):
        """
        Refines the mesh using the given method.

        Parameters
        ----------

        method : str
            A string specifying the refinement method to use
        """
        refine(mesh=self, method=method, inplace=True, **kwargs)

    def search(self, points):
        """
        Determines the containing cell for every given global point
        as well as the corresponding local point on the reference domain.

        Parameters
        ----------

        points : array_like
            The global query points
        """

        containing_cells = self.Geometry._find_cells(points=points,
                                                     cells=self.cells,
                                                     return_cells=True)
        local_points = self.RefMap.eval_inverse(points=points, cells=containing_cells)

        # test if computed local points really are inside reference domain
        eps = 1e-14
        m0 = np.logical_and(local_points[0] > -eps, local_points[1] > -eps)
        m1 = preimages.sum(axis=0) <= 1. + eps
        
        inside = np.logical_and(m0, m1)

        if not inside.all():
            raise RuntimeError('GPS could not find all containing cells and preimages!')

        return containing_cells, local_points
