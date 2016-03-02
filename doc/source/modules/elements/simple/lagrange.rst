.. _modules_elements_simple_lagrange:

:mod:`lagrange` Module
======================

Provides *Lagrange* type elements for simplicial domains
For this type of element each basis function is associated
with a node on the domain where it is equal to one and vanishes
at all other nodes.

:class:`P1` Class
-----------------

.. tikz::
   \draw (0,0) -- (1,0) -- (0,1) -- (0,0);
   \draw[black,fill=black] (0,0) circle (.5ex);
   \draw[black,fill=black] (1,0) circle (.5ex);
   \draw[black,fill=black] (0,1) circle (.5ex);

.. autoclass:: pysofe_light.elements.simple.lagrange.P1
   :members:
   :member-order: bysource

