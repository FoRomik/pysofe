# import everything we need
import pysofe
from pysofe.meshes import Mesh
from pysofe.elements import P1
from pysofe.spaces import FESpace
from pysofe.pde.conditions import DirichletBC
from pysofe.pde import Poisson

from numpy import array, sin, pi, logical_or

# create mesh
nodes = array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
cells = array([[1,2,3],[4,3,2]])
mesh = Mesh(nodes, cells)

# refine the mesh
mesh.refine(times=4)

# create reference element
p1_elem = P1(2)

# create fe space
Vh = FESpace(mesh, p1_elem)

# define boundary conditions
def dirichlet_domain(x):
    dir0 = logical_or(x[0] == 0., x[0] == 1.)
    dir1 = logical_or(x[1] == 0., x[1] == 1.)
    return logical_or(dir0, dir1)

g = 1.

dir_bc = DirichletBC(Vh, dirichlet_domain, g)

# define pde
def f(x):
    return 2. * sin(2.*pi*x[0]) * sin(2.*pi*x[1])

pde = Poisson(Vh, 1, f, dir_bc)

# solve it
U = pde.solve()

# display results
from pysofe.spaces.functions import FEFunction
u = FEFunction(Vh, U)

from IPython import embed as IPS
IPS()
