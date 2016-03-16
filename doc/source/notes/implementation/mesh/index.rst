.. include:: /macros.hrst

.. _notes_impl_mesh:

The Mesh
========

In the finite element environment the purpose of the mesh is to
discretize and approximate the spatial domain :math:`\Omega` of
the considered partial differential equation.

In |PySOFE| the mesh is defined by its *geometry* and a *topology*
implemented in the respective classes |MeshGeometry| and |MeshTopology|
which is a concept similar to the one presented in :cite:`Logg09`.
Furthermore, it stores a family of *reference maps*, implemented in
the |ReferenceMap| class, that connect the physical mesh entities to
a reference domain.

.. toctree::
   :maxdepth: 1

   geometry
   topology
   
	      
      
