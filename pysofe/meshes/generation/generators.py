"""
Provides different mesh generators.
"""

# IMPORTS
import itertools
import numpy as np
from scipy.spatial import Delaunay
from scipy import sparse

from pysofe import utils
from .distance_functions import SignedDistanceFunction

# DEBUGGING
from IPython import embed as IPS
import matplotlib.pyplot as plt

DEBUG = False

class MeshGenerator(object):
    """
    Generic simplicial mesh generator.

    Parameters
    ----------

    sdf : callable
        The signed distance function specifying the mesh geometry

    elf : callable
        The (relative) edge length function
    """

    def __init__(self, sdf, elf=None, **kwargs):
        # set signed distance function and desired edge length function
        self.sdf = sdf

        if elf is not None:
            self.elf = elf
        else:
            self.elf = uniform_edge_lengths

        # set some other method parameters

        # stopping tolerance for changes in nodes movement
        self._stol = kwargs.get('ptol', 1e-2)

        # nodes movement threshold for retriangulation
        self._ttol = kwargs.get('ttol', 1e-1)

        # 'internal' pressure
        self._fscale = kwargs.get('fscale', 1.0)

        # tolerance for geometry evaluation
        self._gtol = kwargs.get('eps', 1e-3)

        # time step size in Euler's method
        self._dt = kwargs.get('dt', 1e-1)

        # step size in numerical differentiation
        self._dx = kwargs.get('dx', np.sqrt(np.finfo(float).eps))

        # we didn't do anything, yet
        self.old_nodes = None
        self.nodes = None
        self.simplices = None
        self.edges = None

    @property
    def stol(self):
        """
        Node movement stopping tolerance relative to average bar lengths.
        """
        return self._stol

    @property
    def ttol(self):
        """
        Node movement retriangulation tolerance.
        """
        return self._ttol

    @property
    def fscale(self):
        """
        The 'internal pressure'.
        """
        return self._fscale

    @property
    def gtol(self):
        """
        Geometry tolerance when evaluating the distance function.
        """
        return self._gtol

    @property
    def dt(self):
        """
        Time step size in Euler's method.
        """
        return self._dt

    @property
    def dx(self):
        """
        Step size in numerical differentiation.
        """
        return self._dx
        
    def _init_distribution(self, bbox, h0):
        """
        Creates an initial distribution of points within the bounding box
        of the geometry.

        First, equally spaced nodes are created and then those outside
        the geometry are removed and those inside are discarded with
        a probability proportional to the density given by the desired
        edge lengths function.

        Parameters
        ----------

        bbox : array_like
            The min/max values of the bounding box in each dimension

        h0 : float
            The initial edge length
        """

        # get the dimension of the bounding box
        dim = np.size(bbox, axis=0)

        # create grid of nodes
        nodes = np.mgrid[tuple(slice(xmin, xmax + h0, h0) for xmin, xmax in bbox)]

        # bring them into the right shape
        nodes = nodes.reshape((dim, -1)).T

        # just keep nodes inside the geometry (allowing some tolerance)
        inside = (self.sdf(nodes.T) < self.gtol)
        nodes = nodes.compress(inside, axis=0)

        # reject nodes with probability proportional to
        # the density 1/(h(x)**2) where h(x) is the edge lengths function
        r = np.random.random(np.size(nodes, axis=0))
        h = self.elf(nodes.T)
        # keep = (r < np.power(h.min()/h, dim))
        d = np.power(h, -dim)
        keep = (r < d/d.max())
        nodes = nodes.compress(keep, axis=0)

        # set current nodes
        self.nodes = nodes

    def _retriangulate(self):
        """
        Generate new simplex indices array by retriangulating
        the current nodes.
        """

        # save node positions
        self.old_nodes = self.nodes.copy()
        
        # retriangulate
        simplices = Delaunay(self.nodes).simplices
        
        # compute centroids of the simplices to just
        # keep interior ones
        centroids = self.nodes.take(simplices, axis=0).mean(axis=1)
        interior = self.sdf(centroids.T) < -self.gtol

        # save new simplices
        self.simplices = simplices.compress(interior, axis=0)

        # and generate new edges
        self._generate_edges()
    
    def _generate_edges(self):
        """
        Generates the index array for the edegs of the current simplices.
        """

        # first we need the number of vertices and their indices
        # of each simplex
        n_vertices = np.size(self.simplices, axis=1)
        indices = range(n_vertices)

        # then we determine all possible combinations
        # of two of those indices
        combs = itertools.combinations(indices, 2)

        # these are the edges
        edges = np.vstack([self.simplices.take(comb, axis=1) for comb in combs])

        # sort them from smaller to bigger index
        edges.sort(axis=1)

        # and remove duplicates
        edges = utils.unique_rows(edges)

        self.edges = edges

    def _compute_forces(self):
        """
        Returns the imaginary spring forces acting on the mesh edges.
        """

        n_nodes, dim = self.nodes.shape
        
        # first we need the lengths of the edges
        edge_nodes = self.nodes.take(self.edges, axis=0)
        edge_vecs = edge_nodes[:,0] - edge_nodes[:,1]
        edge_lengths = np.sqrt(np.power(edge_vecs, 2).sum(axis=1))

        # evaluate the edge length function at the edge midpoints
        # to get the desired relative lengths and multiply them
        # by a scaling factor and a fixed factor to ensure repulsive
        # forces at most edges
        desired_lengths = self.elf(edge_nodes.mean(axis=1).T)

        fscale = self.fscale + 0.4 / 2**(dim - 1)

        tmp = np.power(edge_lengths, dim).sum() / np.power(desired_lengths, dim).sum()
        desired_lengths *= fscale * np.power(tmp, 1./dim)

        # compute the spring forces as the difference between the edge lengths
        # and neglect negative forces
        forces = np.maximum(desired_lengths - edge_lengths, 0)

        # the force resultant is the sum of force vectors, from all edges meeting
        # at a node
        # a stretching force has positive sign, and its direction is given by
        # the two-component vector in edges
        forces_vec = (forces / edge_lengths)[:,None] * edge_vecs

        i = self.edges.take(np.repeat([0,1], dim), axis=1)
        j = np.tile(range(dim), reps=(np.size(forces), 2))
        data = np.hstack([forces_vec, -forces_vec])

        forces = sparse.coo_matrix((data.flat, (i.flat,j.flat)), shape=(n_nodes,dim)).toarray()

        return forces, desired_lengths, edge_lengths

    def _project_back(self, h0):
        """
        Each node outside the geometry is moved back to its closest point
        on the boundary.

        The gradient of the distance function gives the (negative)
        direction to this point and is obtained by numerical differentiation.
        """

        dim = self.nodes.shape[1]
        
        # determine outside nodes (positive distance to boundary)
        dists = self.sdf(self.nodes.T)
        outside = (dists > 0)

        if outside.any():
            # compute gradient of the distance function
            # at outside nodes
            outside_nodes = self.nodes.compress(outside, axis=0)
            outside_dists = dists.compress(outside)

            # steps for numerical integration
            dX = self.dx * h0 * np.identity(dim)

            grads = np.vstack([(self.sdf((outside_nodes + dX[i]).T) - outside_dists) / (self.dx * h0)
                               for i in xrange(dim)])

            # project back
            self.nodes[outside] -= (outside_dists * grads).T

        return dists

    def generate(self, h0, bbox=None, fixed_nodes=None):
        """
        Generates the nodes and cells of the mesh.

        Parameters
        ----------

        bbox : array_like
            The min/max values of the bounding box in each dimension

        h0 : float
            The edge lengths in the initial nodes distribution
        """

        # check bounding box input
        if bbox is None:
            # check if sdf belongs to right class
            if isinstance(self.sdf, SignedDistanceFunction):
                bbox = self.sdf.bbox
            else:
                raise ValueError("Missing bounding box!")
        else:
            if not isinstance(bbox, np.ndarray):
                bbox = np.asarray(bbox)

        assert bbox.ndim == 2
        assert bbox.shape[1] == 2 # min/max values

        # determine dimension
        dim = bbox.shape[0]

        # check fixed nodes input
        if fixed_nodes is not None:
            if not isinstance(fixed_nodes, np.ndarray):
                 fixed_nodes = np.asarray(fixed_nodes)
            
            assert fixed_nodes.ndim == 2

        else:
            fixed_nodes = np.empty(shape=(0, dim))

        n_fixed_nodes = np.size(fixed_nodes, axis=0)

        # first, we create an initial distribution of nodes
        # within the bounding box
        self._init_distribution(bbox, h0)

        # add fixed nodes
        self.nodes = np.vstack([fixed_nodes, self.nodes])
        
        # force retriangulation in first iteration
        self.old_nodes = np.inf
        
        # this is the main loop
        nIt = 0

        if DEBUG:
            fig, ax = plt.subplots()
            fig.show()
        
        while True:
            nIt += 1
            print nIt
            
            # calculate nodes movement
            dists = np.sqrt(np.power(self.nodes - self.old_nodes, 2).sum(axis=1))

            # retriangulate if the max displacement is
            # greater than the tolerance (relative to edge length)
            if dists.max() > self.ttol * h0:
                self._retriangulate()

            # move nodes based on edge lengths and forces
            forces, L, L0 = self._compute_forces()

            # remove nodes that are too close to each other
            # ...

            # set forces at fixed nodes equal to zero
            forces[:n_fixed_nodes] = 0

            # update node positions using Euler's method
            self.nodes += self.dt * forces

            # project nodes that moved outside the geometry
            # back on the boundary
            dists = self._project_back(h0)

            # check termination criterion
            inside = (dists < -self.gtol * h0)
            delta_nodes = np.sqrt(np.sum(self.dt * np.power(forces.compress(inside, axis=0), 2), axis=1))

            if delta_nodes.max() < self.stol * h0:
                break
            elif DEBUG:
                if nIt % 1000 == 0:
                    ax.clear()
                    ax.triplot(self.nodes[:,0], self.nodes[:,1], self.simplices)
                    plt.draw()
                    print "DELTA: ", delta_nodes.max(), "({})".format(self.stol * h0)
                    IPS()

        return self.nodes, self.simplices.astype('int')

def uniform_edge_lengths(points):
    """
    Wrapper for a uniform edge length function.
    """
    return np.ones(np.size(points, axis=1))

