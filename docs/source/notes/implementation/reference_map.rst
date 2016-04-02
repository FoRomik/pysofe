.. include:: /macros.hrst

.. _notes_impl_reference_maps:

The Reference Maps
==================

Within the finite element method there are certain operations that have to
be done on every mesh element. Examples include the evaluation of basis functions
and their derivatives or the evaluation of a function in quadrature points.
Doing this for each mesh entity :math:`K` separately becomes infeasable when
the number of elements increases.

Hence, those operations are rather pulled back on a single *reference element*
:math:`\hat{K}` using a socalled *reference map* :math:`\Phi_{K}:\hat{K}\to K`
for each mesh element :math:`K \in \mathcal{T}_h` where :math:`\mathcal{T}_h`
is our *triangulation*, i.e. the union of all the elements that discretize the
spatial domain of the considered partial differential equation.

.. tikz::

   \draw (4,-1) -- (5,-1) -- (6,-1) -- (6,0) --
          (6,1) -- (5,1) -- (4,1) -- (4,0) -- (4,-1);
   \draw (4,0) -- (5,0) -- (6,0);
   \draw (5,-1) -- (5,0) -- (5,1);
   \draw (5,-1) -- (4,0);
   \draw (6,-1) -- (5,0) -- (4,1);
   \draw (6,0) -- (5,1);
   \draw (0,-0.5) -- (1,-0.5) -- (0,0.5) -- (0,-0.5);

   \node at (0.2,-0.2) {\small $\hat{x}$};
   \node at (0.5,-0.8) {\small $\hat{K}$};
   \draw [->, line width=1pt] (0.3,-0.1) to [out=50, in=130] (4.6,-0.2);
   \node at (2.5,1.0) {\small $\Phi_{K_{7}}$};
   \node at (4.7,-0.4) {\small $x$};
   \node at (5.7,-1.5) {\small $K_{7}$};
   \draw (4.8,-0.6) -- (5.5, -1.3);

Now, every point :math:`x \in K` in some mesh element :math:`K \in \mathcal{T}_h`
can be expressed as

.. math::
   x = \Phi_{K}(\hat{x})

for an appropriate point :math:`\hat{x}\in\hat{K}` on the reference element.
An example where this will be used is numerical integration. Instead of storing
or computing the quadrature points for the domain of every single mesh element
we just have to store the quadrature points for the reference domain and can
map them to each of the physical mesh elements when needed.

Another advantage of this approach is, that we don't have to explicitly define
all the basis functions corresponding to the *degrees of freedom* of the finite
dimensional subspace :math:`V_h` (INSERT LINK TO THEORY). Consider herefore
the above mesh with :math:`9` nodes and suppose we use linear Lagrange basis
functions. In this case the degrees of freedom correspond to the nodes of the
mesh and the basis functions :math:`\mathcal{B} = \{\varphi_j, j=1,\ldots,9\}`
are defined to be equal to :math:`1` at exactly one node :math:`x_k, k=1,\ldots,9`
and vanish at all other nodes, i.e.

.. math::
   :nowrap:
      
   \varphi_j(x_k) = \begin{cases} 1 & j = k \\ 0 & j \not= k \end{cases}

Since we pull back the evaluation of basis functions to the reference element
we only have to define :math:`3` *reference basis functions*
:math:`\hat{\varphi}_i, i=1,2,3` corresponding to the nodes of the reference
element. Then, by means of the reference maps we can evaluate any basis function
:math:`\varphi \in \mathcal{B}` in any point :math:`x \in K` by evaluating the
appropriate reference basis function :math:`\hat{\varphi}` in the corresponding
local point :math:`\hat{x} \in \hat{K}`.
      
.. math::

   \varphi(x)\big|_{x\in K} = \varphi(\Phi_K(\hat{x})) = \hat{\varphi}(\hat{x})

Evaluation of the Reference Maps
--------------------------------

The reference maps are used in many different ways. They have to provide
*zero order information*, i.e. ordinary evaluation, to map local points
on the reference domain to their global counterparts on the physical mesh
elements. *First order information* is needed e.g. to compute Jacobians
arising when we transform integrals to the reference domain or for the
evaluation of derivatives of the global basis functions.

Furthermore it doesn't suffice to be able to map points from the reference
element domain to the mesh elements domains. We also need to be able to
map local points to the mesh subentities, e.g. edges in 2D.

To show how the reference maps are described and evaluated we consider
the 2-dimensional case and straight-sided triangular elements. Then the
reference map :math:`\Phi_K` for :math:`K \in \mathcal{T}_h` can be
defined as

.. math::
   :nowrap:

   \Phi_{K}(\mathbf{\hat{x}}) = \mathbf{c}_1 +
   (\mathbf{c}_2 - \mathbf{c}_1\ \mathbf{c}_3 - \mathbf{c}_1)\cdot\mathbf{\hat{x}}

where :math:`\mathbf{c}_1, \mathbf{c}_2, \mathbf{c}_3 \in\mathbb{R}^2` are
the coordinates of the nodes of the mesh element :math:`K`. An alternative
way of describing this in terms of the local basis functions
:math:`\{ \hat{\varphi}_1, \hat{\varphi}_2, \hat{\varphi}_3 \}` is

.. math::
   :nowrap:

   \Phi_{K}(\mathbf{\hat{x}}) = \mathbf{c}_1 \hat{\varphi}_1(\mathbf{\hat{x}})
                              + \mathbf{c}_2 \hat{\varphi}_2(\mathbf{\hat{x}})
		              + \mathbf{c}_3 \hat{\varphi}_3(\mathbf{\hat{x}})
			      = \sum_{i=1}^{3} \mathbf{c}_i \hat{\varphi}_i(\mathbf{\hat{x}})

Recall herefore that :math:`\hat{\varphi}_i(\mathbf{\hat{c}}_j) = \delta_{ij}`,
where :math:`\{\mathbf{\hat{c}}_j, j=1,2,3\}` are the coordinates of the reference
element's nodes. This means e.g. for :math:`\mathbf{\mathbf{\hat{x}}} = \mathbf{\hat{c}}_1`
that

.. math::

   \hat{\varphi}_1(\mathbf{\hat{c}}_1) = 1
   \qquad \hat{\varphi}_2(\mathbf{\hat{c}}_1) = 0
   \qquad \hat{\varphi}_3(\mathbf{\hat{c}}_1) = 0

so the local node :math:`\mathbf{\hat{c}}_1` is correctly mapped to its global
counterpart

.. math::
   :nowrap:

   \begin{align*}
   \Phi_{K}(\mathbf{\hat{c}}_1) &= \mathbf{c}_1 \hat{\varphi}_1(\mathbf{\hat{c}}_1)
                                 + \mathbf{c}_2 \hat{\varphi}_2(\mathbf{\hat{c}}_1)
		                 + \mathbf{c}_3 \hat{\varphi}_3(\mathbf{\hat{c}}_1) \\
				&= \mathbf{c}_1 \cdot 1 + \mathbf{c}_2 \cdot 0 + \mathbf{c}_3 \cdot 0 \\
				&= \mathbf{c}_1.
   \end{align*}

So the reference map :math:`\Phi_K` for each mesh element :math:`K` is
evaluated by

.. math::

   \Phi_{K}(\mathbf{\hat{x}}) =
       \sum_{i=1}^{nB} \mathbf{c}_i \hat{\varphi}_1(\mathbf{\hat{x}})

where :math:`\mathbf{c}_i, i=1,\ldots,nB` are the coordinates of the mesh
elements nodes and :math:`nB` is the number of local basis functions defined
for the reference element (:math:`2` for edges, :math:`3` for cells in case
of triangles).

In the code all reference maps are evaluated simultaneously for a given point
or even for a given array of local points on the reference domain as follows.

Let :math:`\mathbf{\hat{X}} \in \mathbb{R}^{nP \times nD}` be the array of
local points, so :math:`nP` is the number of points and :math:`nD` is their
dimension (in our case :math:`nD = 2` for local points on the reference triangle).

First we evaluate the local basis functions in those points which will give us
an array :math:`B \in\mathbb{R}^{nB \times nP}` with values between :math:`0`
and :math:`1`

.. math::
   B = \left[\hat{\varphi}_1(\mathbf{\hat{X}})\ \ldots\
             \hat{\varphi}_{nB}(\mathbf{\hat{X}})\right]^{T} \quad\text{with }
   \hat{\varphi}_i(\mathbf{\hat{X}}) \in \mathbb{R}^{nP}, i=1,\ldots,nB.
   
Next, we need the coordinates of the nodes of every mesh element which will
result in an array :math:`C \in \mathbb{R}^{nE \times nB \times nD}`
where :math:`nE` is the number of mesh elements, so

.. math::
   C = [C_1\ \ldots\ C_{nE}] \quad\text{with }
   C_k \in\mathbb{R}^{nB \times nD}, k=1,\ldots,nE.

Then the reference map for each element is evaluated as the matrix product

.. math::

   \Phi_k = B^{T}C_k \in \mathbb{R}^{nP \times nD} \qquad k = 1,\ldots,nE.

which will give us the array of the family of reference maps evaluated in
each given point :math:`\Phi \in \mathbb{R}^{nE \times nP \times nD}`.
