.. include:: /macros.hrst

.. _notes_impl_elem_lagrange:

Lagrange Elements
=================

*Lagrange elements*, also often called *Courant elements*, where first defined
in :cite:`Courant43` with use of Lagrange interpolation polynomials. The basis
functions :math:`\hat{\varphi}_i` for those elements are associated with certain
points, or *nodes*, :math:`\hat{x}_j` on the element's domain. Each basis function
is defined to be equal to :math:`1` at exactly one node and to vanish at all other
nodes, i.e.

.. math::
   :nowrap:
      
   \hat{\varphi}_i(\hat{x}_j) = \delta_{ij}
   = \begin{cases} 1 & i = j \\ 0 & i \not= j \end{cases}

The number of nodes required for the definition of the basis functions is
determined by the their polynomial order.

Linear :math:`\mathbb{P}_1` Elements
------------------------------------

The simplest Lagrange element is the :math:`\mathbb{P}_1` element which defines
*linear basis functions*, each associated with one of the elements vertices.

to be continued...
