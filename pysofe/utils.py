"""
Provides some auxilliary functions that are used in various
points in the software.
"""

# IMPORTS
import numpy as np

# DEBUGGING
from IPython import embed as IPS

def unique_rows(A, return_index=False, return_inverse=False):
    """
    Returns `B, I, J` where `B` is the array of unique rows from
    the input array `A` and `I, J` are arrays satisfying
    `A = B[J,:]` and `B = A[I,:]`.

    Parameters
    ----------

    A : numpy.ndarray
        The 2d array of which to determine the unique rows

    return_index : bool
        Whether to return `I`

    return_inverse : bool
        Whether to return `J`
    """
    
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "Input array must be 2-dimensional"

    B = np.unique(A.view([('', A.dtype)]*A.shape[1]),
                  return_index=return_index,
                  return_inverse=return_inverse)

    if return_index or return_inverse:
        return (B[0].view(A.dtype).reshape((-1, A.shape[1]), order='C'),) \
            + B[1:]
    else:
        return B.view(A.dtype).reshape((-1, A.shape[1]), order='C')

def lagrange_nodes(dimension, order):
    """
    Returns the nodes that determine the Lagrange shape functions of
    given order on a simplicial domain.
    
    Parameters
    ----------

    dimension : int
        The spatial dimension of the points

    order : int
        The polynomial order of the shape functions
    """
    assert dimension in {1, 2, 3}
    assert order >= 0
    
    if dimension == 1:
        if order == 0:
            nodes = np.array([[1/3.]])
        elif order > 0:
            points1d = np.linspace(0., 1., order+1)
            nodes = np.atleast_2d(np.hstack([points1d[0], points1d[-1],
                                             points1d[1:-1]]))
    elif dimension == 2:
        if order == 0:
            nodes = np.array([[1/3.],
                              [1/3.]])
        elif order > 0:
            # 3 vertices
            vertex_nodes = np.array([[0., 1., 0.],
                                     [0., 0., 1.]])
            
            # 3 * (p-1) edge nodes
            points1d = np.linspace(0., 1., (order-1)+2)[1:-1]
            
            edge_nodes_1 = np.vstack([points1d, np.zeros_like(points1d)])
            edge_nodes_2 = np.vstack([np.zeros_like(points1d), points1d])
            edge_nodes_3 = np.vstack([points1d[::-1], points1d])
            
            # (p-1)*(p-1) / 2 interior nodes
            gridx, gridy = np.meshgrid(points1d, points1d)
            grid = np.vstack([gridx.flat, gridy.flat])
            
            interior_nodes = grid.compress(grid.sum(axis=0) < 1, axis=1)
            
            nodes = np.hstack([vertex_nodes,
                               edge_nodes_1, edge_nodes_2, edge_nodes_3,
                               interior_nodes])
    elif dimension == 3:
        if order == 0:
            nodes = np.array([[1/3.],
                              [1/3.],
                              [1/3.]])
        elif order > 0:
            # 4 vertices
            vertex_nodes = np.array([[0., 1., 0., 0.],
                                     [0., 0., 1., 0.],
                                     [0., 0., 0., 1.]])
            
            # 6 * (p-1) edge nodes
            points1d = np.linspace(0., 1., (order-1)+2)[1:-1]
            zeros1d = np.zeros_like(points1d)
            
            edge_nodes_1 = np.vstack([points1d, zeros1d, zeros1d])
            edge_nodes_2 = np.vstack([zeros1d, points1d, zeros1d])
            edge_nodes_3 = np.vstack([zeros1d, zeros1d, points1d])
            edge_nodes_4 = np.vstack([points1d[::-1], points1d, zeros1d])
            edge_nodes_5 = np.vstack([points1d[::-1], zeros1d, points1d])
            edge_nodes_6 = np.vstack([zeros1d, points1d[::-1], points1d])
            
            # (p-1)*(p-2)*(p-3) / 3 interior nodes
            gridx, gridy, gridz = np.meshgrid(points1d, points1d, points1d)
            grid = np.vstack([gridx.flat, gridy.flat, gridz.flat])
            
            interior_nodes = grid.compress(grid.sum(axis=0) < 1, axis=1)
            
            nodes = np.hstack([vertex_nodes,
                               edge_nodes_1, edge_nodes_2, edge_nodes_3,
                               edge_nodes_4, edge_nodes_5, edge_nodes_6,
                               interior_nodes])

    
    return nodes

def match_nodes(nodes0, nodes1, dim=0):
    """
    Determines index sets `I, J` such that
    `nodes0[I, d] == nodes1[J, d] for each d != dim`.
    """

    # check input
    if not nodes0.shape == nodes1.shape:
        err_msg = "Incompatible shapes {} == {}"
        raise ValueError(err_msg.format(nodes0.shape, nodes1.shape))

    if not nodes0.shape[1] > 1:
        raise ValueError("Nodes have to be at least 2-dimensional!")

    # get dimensions to check
    check = [d for d in xrange(np.size(nodes0, axis=1)) if d != dim]

    nodesI = nodes0.take(check, axis=1)
    nodesJ = nodes1.take(check, axis=1)

    I = np.lexsort(nodesI.T)
    J = np.lexsort(nodesJ.T)

    if not np.allclose(nodesI.take(I, axis=0), nodesJ.take(J, axis=0)):
        raise RuntimeError("Cannot match nodes along axis {}".format(axis))

    return I, J

def int2bool(arr, size=None):
    """
    Turns an integer index array into a boolean mask.

    Parameters
    ----------

    arr : array_like
        The 1d array with the indices of the later `True` values

    size : int
        The size of the boolean mask, if `None` the maximum
        integer index will be used
    """

    # make sure input is valid
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype='int')

    assert arr.ndim == 1

    if size is None:
        size = arr.max()
        
    # allocate space
    mask = np.zeros(size, dtype=bool)

    mask[arr] = True

    return mask

def bool2int(mask):
    """
    Returns the indices of the nonzero entries in the given 1D boolean mask.

    Parameters
    ----------

    mask : array_like
        The boolean mask
    """

    # check input
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask, dtype=bool)

    assert mask.ndim == 1
    assert mask.dtype == bool
    
    return mask.nonzero()[0]

def mesh_area(mesh):
    r"""
    Computes the area of a 2D triangular mesh using Heron's formula.

    Given the lengths :math:`a,b,c` of the edges of a triangle
    its area is given by
    
    .. math::
       \sqrt{s(s-a)(s-b)(s.c)}

    where :math:`s = \frac{a + b + c}{2}`.
    """

    # first we need the length of all edges
    edges = mesh.nodes.take(mesh.edges-1, axis=0)
    edge_vecs = np.abs(np.diff(edges, axis=1)[:,0,:])
    edge_lengths = np.sqrt(np.power(edge_vecs, 2).sum(axis=1))

    # then we need the edge lengths w.r.t to each triangle
    inc_2_1 = mesh.topology.get_connectivity(2, 1, return_indices=True)
    abc = edge_lengths.take(inc_2_1 - 1)

    # now we can apply Heron's formula
    s = 0.5 * abc.sum(axis=1)
    A = np.sqrt(s * (s[:,None] - abc).prod(axis=1)).sum()

    return A
