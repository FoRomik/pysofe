import numpy as np
from IPython import embed as IPS

import pysofe_light as pysofe
from pysofe_light.elements import P1
from pysofe_light.meshes import Mesh
from pysofe_light.spaces import FESpace

# create shape element and mesh
p1 = P1(dimension=2)
nodes = np.array([[0.], [1.]])
cells = np.array([[1,2]])

mesh = Mesh(nodes, cells)

# refine the mesh
#mesh.refine(times=1)

# create fe space
#fes = FESpace(mesh, p1)

IPS()
