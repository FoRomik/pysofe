"""
Provides some auxilliary functions that are used in various
points in the software.
"""

import numpy as np

def unique_rows(A, return_index=False, return_inverse=False):
    """
    Returns `B, I, J` where `B` is the array of unique rows from
    the input array `A` and `I, J` are arrays satisfying
    `A = B[J,:]` and `B = A[I,:]`.

    Parameters
    ----------

    A : numpy.ndarray
        The 2d array for which to determine the unique rows

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

def sub_grid_nodes(n):
    '''
    Returns the nodes of the sub grid triangles...???
    
    Parameters
    ----------

    n : int
        Number of grid points on each edge
    '''

    assert n >= 2

    ls = np.linspace(0., 1., n)
    x0, x1 = np.meshgrid(ls,ls)

    nodes = np.vstack([x0.flat, x1.flat])
    nodes = nodes.compress(nodes.sum(axis=0) <= 1., axis=1)

    return nodes
