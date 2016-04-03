.. include:: /macros.hrst

.. _notes_impl_elem_lagrange:

Lagrange Elements
=================

One of the most widely used family of finite elements are the
*Lagrange elements*, also often called *Courant elements*, which were
first defined in :cite:`Courant43` with use of Lagrange interpolation
polynomials. Their defining functionals :math:`N_{i}` are given by

.. math::
   :label: lagrange-fnc
   :nowrap:

   N_i(v) = v(\xi_i),\quad i=1,\ldots,n

where :math:`\xi_i \in\mathbb{R}^d` are specific node points. As each
basis function is therefore associated with a particular node they
form a socalled *nodal basis* of :math:`P(\hat{K})`.

It follows from the equations :eq:`determine-shape-fnc` and
:eq:`lagrange-fnc` that

.. math::
   :nowrap:

   \hat{\varphi}_{i}(\xi_{j}) = \delta_{ij}

for the nodes :math:`\xi_j` on the elements' domain. The number of
nodes required for the definition of the shape or basis functions is
determined by their polynomial order.

:math:`\mathbb{P}_1` Elements
-----------------------------

The Lagrange elements :math:`\mathbb{P}_1` define linear shape
functions on simplicial domains. The nodes :math:`\xi_i` each basis
function is associated with are the vertices of the element.

.. texfigure:: fig_p1_shape_fnc.tex
   :align: center

:math:`\mathbb{P}_2` Elements
-----------------------------

The Lagrange elements :math:`\mathbb{P}_2` define quadratic shape
functions and in addition to the elements' vertices use the midpoints
of the elements' edges as node for their definition.

.. texfigure:: fig_p2_shape_fnc.tex
   :align: center

