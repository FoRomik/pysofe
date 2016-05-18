.. include:: /macros.hrst

.. _notes_impl_elem:

The Reference Element
=====================

In the sense of *Ciarlet* :cite:`Ciarlet78` a *finite element* is
defined as a triple :math:`(\hat{K}, P(\hat{K}), \Sigma(\hat{K}))`
where :math:`\hat{K} \subset \mathbb{R}^d` is a closed bounded subset
with non-empty interior, :math:`P(\hat{K})` is a finite dimensional
vector space on :math:`\hat{K}` and :math:`\Sigma(\hat{K})` is a basis
of the dual space :math:`P'(\hat{K})`, i.e. the space of linear
functionals on :math:`P(\hat{K})`.

In most cases :math:`\hat{K}` is a polygonial domain and
:math:`P(\hat{K})` is a polynomial function space. The elements
:math:`\hat{\varphi}_j, j=1,\ldots,n` of a basis of
:math:`P(\hat{K})`, with :math:`n\in\mathbb{N}` being its dimension,
are called *shape functions* and the linear functionals :math:`N_i
\in\Sigma(\hat{K}), i=1,\ldots,n` define the socalled *degrees of
freedom*.

The shape functions :math:`\hat{\varphi}_j` are determined by the
linear functionals :math:`N_i` via

.. math::
   :nowrap:
   :label: determine-shape-fnc

   N_{i}(\hat{\varphi}_{j}) = \delta_{ij} = \begin{cases}
                                              1 & i = j \\ 
					      0 & \text{otherwise} 
					    \end{cases}

so the choice of these functionals provides different families of
finite elements (see :cite:`Larson13`).

The purpose of the *reference element* is to provide methods for the
evaluation of these shape or basis functions :math:`\hat{\varphi}`
defined on the reference domain and their derivatives.

.. toctree::
   :maxdepth: 2

   lagrange
