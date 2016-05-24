import numpy as np
import pysofe

from IPython import embed as IPS

# create mesh
mesh = pysofe.meshes.UnitSquareMesh(32, 32)

# create reference element
element = pysofe.elements.P2(2)

# create fe space
fes = pysofe.spaces.FESpace(mesh, element)

# dirichlet bc
def dirichlet(x):
    return np.logical_or(x[1] == 0., x[1] == 1.)

dir_bc = pysofe.pde.conditions.DirichletBC(fes, domain=dirichlet, g=0.)

# periodic bc 0
def master0(x):
    return x[0] == 0.

def slave0(x):
    return x[0] == 1.

per_bc0 = pysofe.pde.conditions.PeriodicBC(fes, master0, slave0)

# # periodic bc 1
# def master1(x):
#     return x[1] == 0.

# def slave1(x):
#     return x[1] == 1.

# per_bc1 = pysofe.pde.conditions.PeriodicBC(fes, master1, slave1)

# source term
def source(x):
    dx0 = x[0] - 0.5
    dx1 = x[1] - 0.5
    return x[0] * np.sin(5*np.pi*x[1]) + np.exp(-(dx0*dx0 + dx1*dx1)/0.02)
    
# define pde
pde = pysofe.pde.Poisson(fes, a=1., f=source,
                         bc=(dir_bc, per_bc0))
                         # bc=(per_bc0, per_bc1))

U = pde.solve()

U = per_bc0.recover_slave_dofs(U)
# U = per_bc1.recover_slave_dofs(U)

uh = pysofe.spaces.functions.FEFunction(fes, U)

IPS()

