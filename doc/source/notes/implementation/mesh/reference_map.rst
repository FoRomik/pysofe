.. include:: /macros.hrst

.. _notes_impl_mesh_reference_maps:

The Reference Maps
==================

Within the finite element method there are certain operations that have to
be done on every mesh cell or other mesh entities. Examples include the
evaluation of basis functions and their derivatives or the evaluation of
a function in quadrature points. Doing this for each mesh entity separately
becomes infeasable when the number of entities increases.

Hence, those operations rather are pulled back on a single *reference element*
:math:`\hat{K}` and the results then are mapped to every physical mesh element
:math:`K` using a socalled *reference map* :math:`\Phi_{K}:\hat{K}\to K`.

.. tikz::

   \draw (4,-1) -- (5,-1) -- (6,-1) -- (6,0) --
          (6,1) -- (5,1) -- (4,1) -- (4,0) -- (4,-1);
   \draw (4,0) -- (5,0) -- (6,0);
   \draw (5,-1) -- (5,0) -- (5,1);
   \draw (5,-1) -- (4,0);
   \draw (6,-1) -- (5,0) -- (4,1);
   \draw (6,0) -- (5,1);
   \draw (0,-0.5) -- (1,-0.5) -- (0,0.5) -- (0,-0.5);
   \node at (0.2,-0.2) {\tiny $\hat{K}$};
   \draw [->, line width=1pt] (0.3,-0.1) to [out=50, in=130] (4.6,-0.2);
   \node at (2.5,1.0) {\small $\Phi_{K_{7}}$};
   \node at (4.7,-0.4) {\tiny $K_{7}$};

to be continued...
