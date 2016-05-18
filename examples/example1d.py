# import everything we need
import pysofe
from pysofe.meshes import Mesh
from pysofe.elements import P1
from pysofe.spaces import FESpace
from pysofe.pde.conditions import DirichletBC
from pysofe.pde import Poisson

from numpy import array, sin, pi, logical_or, where

# create mesh
nodes = array([[0.0], [0.25], [0.5], [0.75], [1.0]])
cells = array([[1,2], [2,3], [3,4], [4,5]])
mesh = Mesh(nodes, cells)

# create reference element
p1_elem = P1(dimension=1)

# create fe space
fes = FESpace(mesh, p1_elem)

# define boundary conditions
def dirichlet_domain(x):
    return logical_or(x[0] == 0., x[0] == 1.)

dir_bc = DirichletBC(fes, dirichlet_domain, ud=0.)

# define pde
pde = Poisson(fe_space=fes, a=1, f=1, bc=dir_bc)

# solve it
sol = pde.solve()

# display results
from pysofe.spaces.functions import FEFunction
u = FEFunction(fes, sol)

pysofe.show(u)
raw_input("Press Enter to continue...")

# from IPython import embed as IPS
# IPS()
