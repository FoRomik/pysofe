import numpy as np
from IPython import embed as IPS

import pysofe
from pysofe.elements import P1
from pysofe.meshes import Mesh
from pysofe.spaces import FESpace
from pysofe.spaces.manager import DOFManager

# create shape element and mesh
p1 = P1(dimension=2)
nodes = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
cells = np.array([[1,2,3],[4,3,2]])

mesh = Mesh(nodes, cells)

# refine the mesh
#mesh.refine(times=1)

p1._dof_tuple = (2,2,2)
fe_space = FESpace(mesh, p1)

IPS()
