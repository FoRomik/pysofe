import numpy as np
from IPython import embed as IPS

import pysofe_light as pysofe
from pysofe_light.elements import P1
from pysofe_light.meshes import Mesh
from pysofe_light.spaces import FESpace
from pysofe_light.pde import Poisson, DirichletBC

# create shape element and mesh
p1 = P1(dimension=2)
nodes = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
cells = np.array([[1,2,3],[4,3,2]])

mesh = Mesh(nodes, cells)

# refine the mesh
#mesh.refine(times=1)

# create fe space
fes = FESpace(mesh, p1)

# define bc
def dirichlet_domain(x):
    dir0 = np.logical_or(x[:,0] == 0., x[:,0] == 1.)
    dir1 = np.logical_or(x[:,1] == 0., x[:,1] == 1.)
    return np.logical_or(dir0, dir1)

dir_bc = DirichletBC(fes, dirichlet_domain, ud=0.)

# define pde
pde = Poisson(fe_space=fes,
              a=1.,
              f=1.,
              bc=dir_bc)
IPS()
