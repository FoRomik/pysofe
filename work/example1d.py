# import everything we need
import pysofe
from pysofe.meshes import Mesh
from pysofe.elements import P1
from pysofe.spaces import FESpace
from pysofe.pde.conditions import DirichletBC
from pysofe.pde import Poisson

from numpy import array, sin, pi, logical_or, where

# create mesh
nodes = array([[0.],[0.5],[0.75],[1.]])
cells = array([[1,2],[2,3],[3,4]])
mesh = Mesh(nodes, cells)
mesh.refine(times=2)

# create reference element
p1_elem = P1(dimension=1)

# create fe space
fes = FESpace(mesh, p1_elem)

# define boundary conditions
def dirichlet_domain(x):
    return logical_or(x[0] == 0., x[0] == 1.)

dir_bc = DirichletBC(fes, dirichlet_domain, ud=0.)

# define pde
def p(x):
    return where(x[0] < 0.5, 4., 0.)

pde = Poisson(fe_space=fes, a=1, f=p, bc=dir_bc)

# solve it
sol = pde.solve()

# display results
from pysofe.spaces.functions import FEFunction
u = FEFunction(fes, sol)

from IPython import embed as IPS
IPS()
