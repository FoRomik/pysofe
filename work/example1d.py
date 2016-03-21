# import everything we need
import pysofe
from pysofe import Mesh, P1, FESpace, DirichletBC, Poisson

from numpy import array, sin, pi, where

# create mesh
nodes = array([[0.],[0.5],[0.75],[1.]])
cells = array([[1,2],[2,3],[3,4]])
mesh = Mesh(nodes, cells)

# create reference element
p1_elem = P1(dimension=1)

# create fe space
fes = FESpace(mesh, p1_elem)

# define boundary conditions
def dirichlet_domain(x):
    return logical_or(x[:,0] == 0., x[:,0] == 1.)

dir_bc = DirichletBC(fes, dirichlet_domain, ud=0.)

# define pde
def p(x):
    return where(x[:,0] < 0.5, 4., 0.)

pde = Poisson(fe_space=fes, a=1, f=p, bc=dir_bc)

# # solve it
# sol = pde.solve()

# # display results
# from pysofe import FEFunction
# u = FEFunction(fes, sol)

from IPython import embed as IPS
IPS()
