import numpy as np
from IPython import embed as IPS

from pysofe_light import elements
import pysofe_light as pysofe

# create shape element and mesh
p1 = elements.P1(dimension=2)
n = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
c = np.array([[1,2,3],[4,3,2]])

m = pysofe.meshes.Mesh(n, c)

# refine the mesh
m.refine(times=1)

# create fe space
fes = pysofe.spaces.FESpace(m, p1)

IPS()
