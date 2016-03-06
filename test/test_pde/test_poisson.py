"""
Tests the Poisson equation.
"""

import numpy as np
import pysofe_light as pysofe

# create neccessary setup
nodes_2d = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
cells_2d = np.array([[1, 2, 3], [2, 3, 4]])
mesh_2d = pysofe.meshes.mesh.Mesh(nodes_2d, cells_2d)
mesh_2d.refine(method='uniform', times=4)

p1_elem = pysofe.elements.simple.lagrange.P1(dimension=2)

fes = pysofe.spaces.space.FESpace(mesh_2d, p1_elem)

class TestPoisson2D(object):
    def dir_domain(x):
        dir0 = np.logical_or(x[0] == 0., x[0] == 1.)
        dir1 = np.logical_or(x[1] == 0., x[1] == 1.)
        return np.logical_or(dir0, dir1)

    dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes,
                                               domain=dir_domain,
                                               ud=1.)

    pde = pysofe.pde.poisson.Poisson(fe_space=fes, a=1., f=0., bc=dir_bc)

    def test_solution(self):
        sol = self.pde.solve()

        assert np.allclose(sol, np.ones(self.pde.fe_space.n_dof))
