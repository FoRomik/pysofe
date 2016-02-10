"""
Provides the data structure that supplies topological information on meshes.
"""

# IMPORTS
import numpy as np

class MeshTopology(object):
    """
    Stores topological information on a mesh such as its
    entities and connectivity information.

    Parameters
    ----------

    cells : array_like
        The connectivity array of the mesh cells

    dim : int
        The spatial dimension of the mesh cells
    """

    def __init__(self, cells, dim):
        # make sure cells array has correct type
        cells = np.asarray(cells, dtype='int')

        # check input
        if not cells.ndim == 2:
            msg = "Cells array has to 2-dimensional storing the cells row-wise!"
            raise ValueError(msg)

        # by now, mesh topology can only handle simplicial elements
        if not np.size(cells, axis=1) == dim + 1:
            raise NotImplementedError('Currently only supports simplicial meshes!')
        else:
            self._dimension = dim
            self._n_vertices = dim + 1

        # the following dictionary is used to store every incidence relation
        # of the mesh entities once they have been computed to avoid recomputation
        self._incidence = dict.fromkeys(range(dim + 1))
        for i in xrange(dim + 1):
            self._incidence[i] = dict.fromkeys(range(dim + 1))

        # initialize incidence relations
        self._init_incidence(cells)

    def _init_incidence(self, cells):
        """
        Computes the incidence relations D -> 0 and 0 -> D 
        from given connectivity array where D is the topological 
        of the mesh cells dimension.

        Parameters
        ----------

        cells : numpy.ndarray
            The connectivity array of the mesh cells
        """

        # get number of cells and their number of vertices
        ncells, nvertices = cells.shape
        
        # the incidence relations are stored as sparse boolean matrices
        #
        # to setup the sparse matrix in coo format we need
        # for every entry `d` a tuple `(r,c)` that specifies
        # its row and column index

        # the entries `d` are simply all 1 because we construct
        # a boolean matrix
        data = np.ones(ncells * nvertices, dtype=bool)

        # the row index `r` of each tuple represents the index of the
        # mesh entity for which we are seeking the incident entities
        # represented by the columns

        # as we are constructing a matrix that shows which vertices are
        # incident to every cell (D->0) the row indices represent the mesh cells
        # and we need each of the `ncells` indices nvertices` times 
        rowind = np.arange(ncells).repeat(repeats=nvertices)

        # each row of the given `cells` array consists of the `nvertices` mesh
        # vertices that are incident to the mesh cell of the repective row index
        # so we can pair each cell with its vertex indices by concatenating
        # the `cells` array row-wise
        # (substraction is needed because indexing starts at `0`)
        colind = cells.ravel(order='C') - 1

        # now we construct our sparse matrix for the incidence relation (D->0)
        D_to_O = sparse.coo_matrix((data, (rowind, colind)))

        # now, we simply have to transpose this matrix to get the relation (0->D),
        # i.e. where the non-zero column indices represent the the mesh cells that
        # are incident to the mesh vertices specified by the row indices
        O_to_D = D_to_O.T

        # we store the incidence matrices in lil format to have access to
        # its useful class property `rows` that returns the index of the non-zero
        # column entries for each row
        self._incidence[self._dimension][0] = D_to_O.tolil()
        self._incidence[0][self._dimension] = O_to_D.tolil()
        
