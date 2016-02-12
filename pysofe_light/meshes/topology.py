"""
Provides the data structure that supplies topological information on meshes.
"""

# IMPORTS
import numpy as np
from scipy import sparse
import itertools

# DEBUGGING
from IPython import embed as IPS

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
            self._n_vertices = np.arange(self._dimension+1) + 1

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

        # the number of vertices derived from the given cells has to
        # comply with the mesh topology's number of vertices of its cells
        assert nvertices == self._n_vertices[-1]
        
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
        
    def get_connectivity(self, d, dd, return_indices=False):
        """
        Returns the incidence relation `d -> dd`, 
        i.e. for each each `d`-dimensional mesh entity its
        incident entities of topological dimension `dd`.
        
        Parameters
        ----------

        d, dd : int
            The topological dimensions of the entities involved

        return_indices : bool
            Whether to return the indices of the incident entities
            instead of the sparse incidence matrix
        """

        if self._incidence[d][dd] is None:
            self._compute_connectivity(d, dd)

        if return_indices:
            return self._get_indices(d, dd)
        else:
            return self._incidence[d][dd]

    def get_entities(self, d):
        """
        Returns the array of all `d`-dimensional mesh entities
        represented by the indices of their incident vertices.

        Parameters
        ----------

        d : int
            The topological dimension of the mesh entities to return
        """

        entities = self._get_indices(d=d, dd=0)

        return entities
        
    def get_boundary(self, d):
        """
        Returns the boundary entities of topological dimension `d`.

        Parameters
        ----------

        d : int
            The topological dimension of the desired boundary entities
        """

        # to determine the boundary we need the incidence relation
        # of the mesh facets to the mesh cells
        D = self._dimension
        incidence_facets_cells = self.get_connectivity(D-1, D)

        # because the boundary facets are incident the only one cell
        boundary_facets = (incidence_facets_cells.sum(axis=1) == 1)

        if d == D-1:
            # if the boundary facets are wanted, we're done
            boundary = np.asarray(boundary_facets).ravel()
        else:
            # else we need the incidence relation of the desired
            # mesh entities to the mesh facets
            incidence_d_facets = self.get_connectivity(d, D-1)

            # and intersect this with the boundary facets
            boundary_entities = incidence_d_facets.dot(boundary_facets)

            boundary = np.asarray(boundary_entities).ravel()

        return boundary
        
    def _compute_connectivity(self, d, dd):
        """
        Computes the incidence relation `d -> dd` using a combination
        of the method `build`, `transpose` and `intersection`.

        Parameters
        ----------

        d, dd: int
            The topological dimensions of the entities involved
        """

        # we need at least the incidence relations `d -> 0` and `dd -> 0`
        # so build them if neccessary
        if self._incidence[d][0] is None:
            self._build(d=d)

        if self._incidence[dd][0] is None:
            self._build(d=dd)

        # check if we even have to compute anything
        if self._incidence[d][dd] is not None:
            return

        # now get down to business
        if d < dd:
            # the methods involved work from higher to lower
            # spatial dimensions so we first go the other way
            self._compute_connectivity(dd, d)

            # and then just transpose
            self._transpose(d, dd)
        else:
            # this should not be the case here
            assert not (d == 0 and dd == 0)
            
            # the computation always works the same way
            # first get `d -> 0` and `0 -> dd` and then
            # just intersect
            self._compute_connectivity(d, 0)
            self._compute_connectivity(0, dd)
            self._intersection(d, dd, 0)
        
    def _get_indices(self, d, dd):
        """
        Returns for every `d`-dimensional mesh entity its incident
        entities of topological dimension `dd`

        Parameters
        ----------

        d, dd: int
            The topological dimensions of the entities involved
        """

        if not d > dd:
            raise ValueError("Returning indices only supported for downward incdence, yet!")

        if self._incidence[d][dd] is None:
            self._compute_connectivity(d, dd)
        
        incidence = self._incidence[d][dd]
        indices = np.asarray(incidence.rows.tolist(), dtype='int') + 1

        return indices
        
    def _build(self, d):
        """
        Computes the incidence relation `d -> 0` for `0 < d < D`
        where `D` is the maximum topological dimension of the mesh entities.

        Parameters
        ----------

        d : int
            The topological dimension of the entities for which to
            build the relation `d -> 0`
        """

        if d == self._dimension:
            # already taken care of by `_init_incidence`
            return
        
        # make sure we can build the relation, i.e. we already have the
        # relation `D -> 0`
        D = self._dimension
        assert self._incidence[D][0] is not None

        # first we need to get the vertex sets that define the
        # `d` dimensional entities of each cell
        vertex_sets = self._local_vertex_sets(d=d)

        # but we need to remove duplicates
        entity_vertices = unique_rows(np.vstack(vertex_sets))

        # now we have every `d`-dimensional mesh entity defined
        # by their vertex indices so we can construct the incidence
        # matrix for the relation `d->0`

        # the entries of the matrix are all 1 and for every
        # entity we need `n_vertices[d]` of them
        n_entities = entity_vertices.shape[0]
        n_vertices = self._n_vertices[d]
        
        data = np.ones(n_entities * n_vertices, dtype=bool)

        # the row indices represent the `d` dimensional mesh entities
        # and as each entity is defined by `n_vertices[d]` vertex indices
        # we need every row index that many times
        rowind = np.arange(n_entities).repeat(n_vertices)

        # the nonzero column indices, which in this case represent the vertices
        # that are incident to the `d` dimensional entities can be taken
        # row-wise from the computed `entity_vertices` array
        colind = entity_vertices.ravel(order='C') - 1

        incidence_d_0 = sparse.coo_matrix((data, (rowind, colind)))

        self._incidence[d][0] = incidence_d_0.tolil()

    def _transpose(self, d, dd):
        """
        Computes the incidence relation `d -> dd` from `dd -> d`
        for d < dd.

        Parameters
        ----------

        d,dd : int
            The topological dimension of the mesh entities involved
        """

        assert d < dd

        if self._incidence[dd][d] is not None:
            if self._incidence[d][dd] is None:
                incidence_dd_d = self.get_connectivity(of=dd, to=d)
                self._incidence[d][dd] = incidence_dd_d.T
            else:
                # incidence relation already exists
                pass
        else:
            msg = 'Incidence ({})->({}) is not available for transposing!'
            raise RuntimeError(msg.format(dd, d))

    def _intersection(self, d, dd, ddd):
        """
        Computes the incidence relation `d -> dd` from `d -> ddd` and `ddd -> dd`
        for d >= dd.

        Parameters
        ----------

        d, dd, ddd: int
            The topological dimensions of the entities involved
        """

        assert d >= dd

        # we compute this intersection via the dot product of the two
        # incidence matrices for the relations `d -> ddd` and `ddd -> dd`
        # so we transfer them into a format more suitable for this
        # (and we need them as integer matrices)
        incidence_d_ddd = self.get_connectivity(of=d, to=ddd).tocsr().astype('int')
        incidence_ddd_dd = self.get_connectivity(of=ddd, to=dd).tocsr().astype('int')

        intersection = incidence_d_ddd.dot(incidence_ddd_dd)

        if d == dd:
            # TODO: explain this and find alternative not involving .toarray()
            intersection = np.mod(intersection.toarray(), self._n_vertices[d])

            incidence_d_dd = sparse.lil_matrix(intersection.astype(bool))
        elif d > dd:
            # in the simplicial case for `d > dd` the `d`-dimensional entities
            # have `dd+1` subentities of topological dimension `dd`
            # TODO: dissolve this restriction of simplicial case
            n_subentities = dd + 1

            # so the `dd`-dimensional entities that are incident with
            # the `d`-dimensional ones are those where the two incidence
            # matrices intersect `n_subentities` times
            incidence_d_dd = (intersection == n_subentities).astype(bool).tolil()
        else:
            raise RuntimeError("You shouldn't have gotten this far...?!")
        
        self._incidence[d][dd] = incidence_d_dd
            
    def _local_vertex_sets(self, d):
        """
        Computes the set of vertex sets incident to the `d`-dimensional
        mesh entities of every mesh cells.

        For example, if the mesh cells are triangles and `d = 1`, then for
        each cell the pairs of indices that define its edges would be computed.

        Parameters
        ----------

        d : int
            The topological dimension of the mesh entities for which to
            compute the vertex sets
        """

        # first we need for each cell its defining vertex indices
        cell_vertices = self._get_indices(d=self._dimension)

        # then we need to know the possible local index combinations
        # that would define the `d`-dimensional mesh entities of each cell
        combs = itertools.combinations(range(self._dimension+1), self._n_vertices[-1])
        # e.g. for a triangle and edges (`d=1`) this would be `[(0,1), (0,2), (1,2)]`
        # because the three edges of a triangle are defined by the
        # first-second (0,1), first-third (0,2) and second-third (1,2)  
        # vertex of that triangle

        # now that we have that possible combinations we can simply take the
        # respective vertex indices from the cell vertices array
        vertex_sets = cell_vertices.take(list(combs), axis=1)

        return vertex_sets

    def _reset(self, cells=None):
        """
        Empties all computed incidence relations and initializes
        new ones if desired.

        Parameters
        ----------

        cells : numpy.ndarray, optional
            The connectivity array of the new mesh cells
        """

        D = self._dimension
        for i in xrange(D+1):
            for j in xrange(D+1):
                self._incidence[i][j] = None

        if cells is not None:
            self._init_incidence(cells)
