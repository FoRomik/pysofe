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
        return elf.topology.get_entities(d=self.dimension - 1)

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

    def refine(self, method='uniform', **kwargs):
        """
        Refines the mesh using the given method.

        Parameters
        ----------

        method : str
            A string specifying the refinement method to use
        """
        refinements.refine(mesh=self, method=method, inplace=True, **kwargs)
