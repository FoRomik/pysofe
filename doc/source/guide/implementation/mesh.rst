.. include:: /macros.hrst

.. _guide_impl_mesh:

The Mesh
========

In the finite element environment the purpose of the mesh is to
discretize and approximate the spatial domain :math:`\Omega` of
the considered partial differential equation.

In |PySOFE| the mesh is defined by its *geometry* and a *topology*
implemented in the respective classes |MeshGeometry| and |MeshTopology|.
This is a concept idea similar to the one presented in :cite:`Logg09`.
Furthermore, it stores a family of *reference maps*, implemented in
the |ReferenceMap| class, that connect the physical mesh entities to
a reference domain.

The Mesh Geometry
-----------------

The |MeshGeometry| class provides geometrical information about the mesh
which currently amounts in storing the spatial coordinate of the mesh nodes.

The Mesh Topology
-----------------

The |MeshTopology| class provides topological information about the mesh, i.e.
it gives access to its entities such as edges, faces or cells and neighborly
relation, e.g. the neighboring cells to each edge.

All the information is stored using *incidence matrices*. So, for every pair
:math:`(d,d')` of topological dimensions :math:`0 \leq d,d' \leq D` where
:math:`D \in \{ 1,2,3 \}` is the spatial dimension of the mesh, there exists
a matrix :math:`M_{d,d'}` that for the :math:`d`\ -dimensional entities stores
their incident entities of topological dimension :math:`d'`. The entries
:math:`m_{ij}` are either :math:`1` or :math:`0` depending on whether the
:math:`j`\ -th :math:`d'`\ -dimensional mesh entity is incident to the
:math:`i`\ -th mesh entity of topological dimension :math:`d`.

To explain how to get to these matrices we will consider the following simple
sample mesh in 2D.

.. tikz:: 
   
   \coordinate [label={below left:{\small $1$}}] (1) at (0,0);
   \coordinate [label={below right:{\small $2$}}] (2) at (2,0);
   \coordinate [label={above left:{\small $3$}}] (3) at (0,2);
   \coordinate [label={above right:{\small $4$}}] (4) at (2,2);

   \draw (3) -- (1) -- (2) -- (3) -- (4) -- (2);
   \draw (0.6,0.6) node[circle, draw, inner sep=1pt, minimum size=0pt] {\small 1};
   \draw (1.4,1.4) node[circle, draw, inner sep=1pt, minimum size=0pt] {\small 2};

which consists of 4 vertices and 2 triangular cells. For this mesh the incidence
relation :math:`D \to 0`, i.e. the vertices incident to each cell :math:`(D=2)`,
would be stored in a matrix

.. math::
   :nowrap:
   :label: sample-mesh
      
   I_{2,0} = \begin{pmatrix}
               1 & 1 & 1 & 0 \\
	       0 & 1 & 1 & 1
	     \end{pmatrix}

This matrix shows that e.g. the first cell, which corresponds to the first
row in the matrix, is incident to the vertices with indices :math:`1,2,3`, as
those are the non-zero column indices which represent the mesh vertices.

Note that this mesh is very simple and coarse. If the meshes get finer the
number of non-zero entries will become very small compared to the total number
of entries, i.e. the incidence matrices will be sparse. To make use of this fact
we will utilize the sparse matrix capabilities of the :py:mod:`scipy <scipy.sparse>`
package which provides sparse matrix classes that only store the non-zero
entries, hence saving lots of memory.

Similarly to :cite:`Logg09` the construction of these matrices relies on the
three algorithms *build*, *transpose* and *intersection*, which will be explained
in the following sections.

Initialize
----------

A prerequisite for the algorithms to work is, that we already have the incidence
matrix for the relation :math:`D \to 0`. This matrix is constructed using the
following auxiliary intialization method.

First, we need the *conenctivity array* that defines the mesh cells (entities of
topological dimension :math:`D`) via their node indices. For the sample mesh
:eq:`sample-mesh` this array would be

.. math::
   :nowrap:

   \begin{bmatrix}
     1 & 2 & 3 \\
     2 & 3 & 4
   \end{bmatrix},

i.e. the first cell is define by the node indices :math:`1,2,3` and the second
cell is defined by the nodes :math:`2,3,4`. This array has to be supplied by
the user and passed as an argument when creating an instance of the |MeshTopology|
class.

In the end we will represent the incidence relation using a sparse matrix in
*coordinate format (COO)* which stores its non-zero entries by their 'cordinates'
:math:`(i,j)`, i.e. their row and column indices. To do so, three 1-dimensional
arrays are used. The first one is the *data* array which contains the values of
the non-zero entries. In our case this will be an array of :math:`1`\ 's.
The second array will contain the row indices and the third one the respective
column indices.

To create the row index array we need to know the number of vertices that define
the cells of the mesh. For a mesh of triangles this number is equal to
:math:`nV = 3` (in 3D for tetrahedra it would be :math:`nV = 4`). Remember
that we want to construct the incidence relation between the mesh cells and the
mesh vertices, so if each cell is defined by :math:`nV` vertices there will be
that many non-zero entries in every row of the matrix. Hence, for each row index
:math:`i` (which corresponds to one cell in the mesh) there will be :math:`nV`
coordinate tuples :math:`(i,j)` that have this index as a first component.
Therefore, the number of vertices tells us how often each cell index appears
in the row array.

For our sample mesh the row index array would be:

.. math::
   :nowrap:

   rows = \begin{bmatrix} 1 & 1 & 1 & 2 & 2 & 2 & 3 & 3 & 3 & 4 & 4 & 4 \end{bmatrix},

containing each row index :math:`nV = 3` times.

The columns of the incidence matrix we create correspond to the vertices of
the mesh. So, the column array will contain the indices of the vertices that
are incident to ...

Build
-----

The *build* algorithm computes the incidence relation :math:`d \to 0` for a
topological dimension :math:`0 < d < D`, i.e. it determines the incident
vertices for every :math:`d`\ -dimensional mesh entity.

