import numpy as np
from IPython import embed as IPS

import pysofe_light as pysofe
from pysofe_light.elements import P1
from pysofe_light.meshes import Mesh
from pysofe_light.spaces import FESpace
from pysofe_light.spaces.manager import DOFManager

# create shape element and mesh
p1 = P1(dimension=2)
nodes = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
cells = np.array([[1,2,3],[4,3,2]])

mesh = Mesh(nodes, cells)

# refine the mesh
#mesh.refine(times=1)

p1._dof_tuple = (1,2,1)
dm = DOFManager(mesh, p1)

IPS()
