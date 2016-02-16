"""
Provides the data structure for the degrees of freedom manager.
"""

# IMPORTS
import numpy as np

class DOFManager(object):
    """
    Connects the finite element mesh and reference element
    through the assignment of degrees of freedom to individual elements.

    Parameters
    ----------

    mesh : pysofe.meshes.Mesh
        The mesh for which to provide connectivity information

    element : pysofe.elements.base.Element
        The reference element
    """

    def __init__(self, mesh, element):
        self._mesh = mesh
        self._element = element

    def get_connectivity_array(self, d):
        """
        Returns the array connecting the local and global degrees of freedom.
        
        Parameters
        ----------

        d : int
            The topological dimension of the mesh entities for which to return
            the connectivity array
        """

        dof_maps = self._compute_connectivity_array()
        
        return dof_maps[d]

    def get_n_dof(self):
        """
        Returns total number of degrees of freedom.
        """

        n_dof = np.abs(self.get_connectivity_array(codim=0)).max()

        return n_dof

    def _compute_connectivity_array(self):
        """
        Establishes the connection between the local and global degrees of freedom
        via the connectivity array `C` where `C[i,j] = k` connects the `i`-th
        global basis function on the `j`-th element to the `k`-th local basis 
        function on the reference element.
        """

        # the assignement of the degrees of freedom from 1 to the total number
        # of dofs will be done according to the following order:
        # 1.) components (scalar- or vector-valued)
        # 2.) entities
        # 3.) dimension (vertices, edges, cells)

        # first the assignment of new dofs
        global_dim = self.mesh.dimension
        n_dofs = 0

        dofs = [None] * (global_dim + 1)
        for dim in xrange(global_dim + 1):
            n_entities = self.mesh.Topology.get_entities(d=dim).shape[0]
            used = self.element.dof_tuple[dim] * n_entities

            dofs[dim] = np.reshape(n_dofs + np.arange(used, dtype=int) + 1, (-1, n_entities))
            n_dofs += used

        # assemble dof maps
        DM = [None] * (global_dim + 1)
        for dim in xrange(global_dim + 1):
            codim = global_dim - dim
            DM[codim] = [None] * (dim + 1)
            for d in xrange(dim + 1):
                if d == dim:
                    n_entities = n_sub_entities = self.mesh.Topology.get_entities(d=d).shape[0]
                    connectivity = np.arange(n_entities, dtype=int)[:,None] + 1
                else:
                    connectivity = self.mesh.Topology.get_connectivity(d=dim, dd=d, return_indices=True)
                    n_entities, n_sub_entities = connectivity.shape
                
                n_dof_loc = dofs[d].shape[0]

                DM[codim][d] = dofs[d].take(connectivity.T - 1, axis=1)
                
                # fix orientation
                if self.mesh._is_sofe_compatible:
                    if dim > d and d == 1:
                        # fix edge orientation
                        sign = self.mesh._orientation(d=1)
                        if self.element.is_nodal:
                            neg_sign = (sign.T < 0)
                            DM[codim][d][:, neg_sign] = DM[codim][d][::-1, neg_sign]
                        elif self.element.is_hierarchic:
                            DM[codim][d][1:n_dof_loc:2] *= sign.T
                        else:
                            raise ValueError('Invalid element type.')
                    elif dim > d and d == 2:
                        # fix face orientation
                        raise NotImplementedError()

                DM[codim][d] = DM[codim][d].reshape((-1, n_entities))

            DM[codim] = np.vstack(DM[codim])

        #self._connectivity_arrays = DM
        return DM
