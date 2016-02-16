"""
Provides methods for the uniform refinement of mesh cells.
"""

# IMPORTS
import numpy as np

def uniform_refine_simplices(mesh):
    """
    Wrapper function to call appropriate method w.r.t.
    the maximum topological dimension of the mesh entities.

    Parameters
    ----------

    mesh : pysofe.meshes.Mesh
        The finite element mesh to refine
    """

    if mesh.dimension == 1:
        return _uniform_refine_intervals(mesh)
    elif mesh.dimension == 2:
        return _uniform_refine_triangles(mesh)
    elif mesh.dimension == 3:
        return _uniform_refine_tetrahedrons(mesh)
    else:
        raise ValueError("Invalid mesh dimension! ({})".format(mesh.dimension))

def _uniform_refine_intervals(mesh):
    """
    Uniformly refines each interval of a 1d mesh into two new subintervals.
    """
    raise NotImplementedError()

def _uniform_refine_triangles(mesh):
    """
    Uniformly refines each triangle of a 2d mesh into four new triangles.
    """

    # get current cells and edges
    cells = mesh.cells
    edges = mesh.edges

    assert np.size(cells, axis=1) == 3

    if 0:
        if not mesh._is_sofe_compatible:
            # but we want the cell vertices to be ordered counterclockwise
            # so we swap two vertices in each cell where they are clockwise
            cw_mask = mesh.topology._cw_faces(nodes=mesh.nodes)
            cells[cw_mask] = cells[cw_mask][:,[0,2,1]]

    # get number of current nodes
    n_nodes = np.size(mesh.nodes, axis=0)
    
    # first we create the new nodes as midpoints of the current edges
    midpoints = 0.5 * mesh.nodes.take(edges - 1, axis=0).sum(axis=1)

    new_nodes = np.vstack([mesh.nodes, midpoints])
    
    # then we generate the indices of the newly created nodes
    #
    # their indices start at the current number of nodes (`n_nodes`) + 1
    # and end after additional `n_edges` nodes
    n_edges = np.size(edges, axis=0)
    new_node_indices = np.arange(n_nodes + 1, n_nodes + n_edges + 1, dtype=int)

    # refine elements
    #
    # for every element we need the indices of the edges at which the corresponding
    # new nodes are created
    indices_2_1 = mesh.topology.get_connectivity(2, 1, return_indices=True) - 1

    if 0:
        if not mesh._is_sofe_compatible:
            # we have to reorder the indices such that they fit to the counterclockwise
            # ordering of the mesh cells (keep in mind that originally the cell vertices
            # are ordered increasingly)
            #
            # TODO: explain the following!

            # for those cells whose vertices where ordered clockwise we have to rotate
            # the edge indices in a clockwise fashion
            cw_mask = mesh.topology._cw_faces(nodes=mesh.nodes)
            indices_2_1[cw_mask] = indices_2_1[cw_mask][:,[1,2,0]]
            
            # and for those whose vertices where already ordered counterclockwise
            # we just have to swap the second and third index
            #ccw_mask = mesh._ccw_faces
            ccw_mask = np.logical_not(cw_mask)
            indices_2_1[ccw_mask] = indices_2_1[ccw_mask][:,[0,2,1]]    
    
    # next we augment the indices that define each element as if
    # they were defined by 6 nodes (including those of the edge midpoints)
    cells_ = np.hstack([cells, new_node_indices.take(indices_2_1, axis=0)])
    
    # now we can generate the four new elements for each existing one
    if 0:
        new_cells_1 = np.vstack([cells_[:,0], cells_[:,3], cells_[:,5]]).T
        new_cells_2 = np.vstack([cells_[:,3], cells_[:,1], cells_[:,4]]).T
        new_cells_3 = np.vstack([cells_[:,5], cells_[:,4], cells_[:,2]]).T
        new_cells_4 = np.vstack([cells_[:,4], cells_[:,5], cells_[:,3]]).T
    else:
        new_cells_1 = np.vstack([cells_[:,0], cells_[:,3], cells_[:,4]]).T
        new_cells_2 = np.vstack([cells_[:,3], cells_[:,1], cells_[:,5]]).T
        new_cells_3 = np.vstack([cells_[:,4], cells_[:,5], cells_[:,2]]).T
        new_cells_4 = np.vstack([cells_[:,5], cells_[:,4], cells_[:,3]]).T
        
    new_cells = np.vstack([new_cells_1, new_cells_2, new_cells_3, new_cells_4])

    return new_nodes, new_cells
