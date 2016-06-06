.. include:: /macros.hrst

.. _notes_impl_mesh_generation:

Mesh Generation
===============

One of the first steps in the practical application of the finite
element method is creating a mesh to discretize the domain of the
considered differential equation. Every subsequent step of the method
strongly depends on the quality of the generated mesh. It is therefore
indispensable to use a mesh of good quality to get an accurate
solution.

A mesh consists of node positions and connectivity information. While
it is possible for simple geometries like a square to explicitly
define or generate the nodes and cells this approach becomes
infeasible once the geometries get more complicated.

In |PySOFE| we implement a method proposed by `Per-Olof Persson
<http://persson.berkeley.edu/>`_ and `Gilbert Strang
<http://www-math.mit.edu/~gs>`_ in their paper :cite:`Persson04`.
They make use of the analogy between a simplicial mesh and a truss
structure and by assuming a force-displacement function for the bars
of the structure, i.e. the edges of the mesh, iteratively generate
meshes by solving for a force equilibrium.

The Algorithm
-------------

to be continued...
