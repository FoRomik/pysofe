"""
Tests the boundary conditions.
"""

# IMPORTS
import numpy as np
from scipy import sparse
import pysofe

# create necessary setup
mesh_1d = pysofe.meshes.Mesh(nodes=np.array([[0.0], [0.3], [0.6], [1.0]]),
                             connectivity=np.array([[1,2], [2,3], [3,4]]))

mesh_2d = pysofe.meshes.Mesh(nodes=np.array([[0.0, 0.0],
                                             [1.0, 0.0],
                                             [0.0, 1.0],
                                             [1.0, 1.0],
                                             [0.5, 0.5]]),
                             connectivity=np.array([[1, 2, 5],
                                                    [1, 3, 5],
                                                    [2, 4, 5],
                                                    [3, 4, 5]]))

element_1d = pysofe.elements.simple.lagrange.P1(dimension=1)
element_2d = pysofe.elements.simple.lagrange.P1(dimension=2)

fes_1d = pysofe.spaces.FESpace(mesh_1d, element_1d)
fes_2d = pysofe.spaces.FESpace(mesh_2d, element_2d)



class TestDirichletBC(object):

    def test_1d_both_ends_fixed(self):
        def dir_domain(x):
            dir0 = np.logical_or(x[0] == 0, x[0] == 1)
            return dir0
        
        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_1d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((4,4))
        b = np.zeros((4,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[1, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 1]]))
        assert np.allclose(b, np.array([[1],
                                        [0],
                                        [0],
                                        [1]]))

    def test_1d_start_fixed(self):
        def dir_domain(x):
            dir0 = (x[0] == 0)
            return dir0
        
        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_1d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((4,4))
        b = np.zeros((4,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[1, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]))
        assert np.allclose(b, np.array([[1],
                                        [0],
                                        [0],
                                        [0]]))

    def test_1d_end_fixed(self):
        def dir_domain(x):
            dir0 = (x[0] == 1)
            return dir0
        
        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_1d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((4,4))
        b = np.zeros((4,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 1]]))
        assert np.allclose(b, np.array([[0],
                                        [0],
                                        [0],
                                        [1]]))

    def test_2d_all_sides_fixed(self):
        def dir_domain(x):
            dir0 = np.logical_or(x[0] == 0., x[0] == 1.)
            dir1 = np.logical_or(x[1] == 0., x[1] == 1.)
            return np.logical_or(dir0, dir1)

        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_2d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[1, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0],
                                                  [0, 0, 1, 0, 0],
                                                  [0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [1],
                                        [1],
                                        [1],
                                        [0]]))

    def test_2d_left_right_fixed(self):
        def dir_domain(x):
            dir0 = np.logical_or(x[0] == 0., x[0] == 1.)
            return dir0

        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_2d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[1, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0],
                                                  [0, 0, 1, 0, 0],
                                                  [0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [1],
                                        [1],
                                        [1],
                                        [0]]))

    def test_2d_top_bottom_fixed(self):
        def dir_domain(x):
            dir1 = np.logical_or(x[1] == 0., x[1] == 1.)
            return dir1

        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_2d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[1, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0],
                                                  [0, 0, 1, 0, 0],
                                                  [0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [1],
                                        [1],
                                        [1],
                                        [0]]))

    def test_2d_left_fixed(self):
        def dir_domain(x):
            dir00 = (x[0] == 0)
            return dir00

        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_2d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[1, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [0],
                                        [1],
                                        [0],
                                        [0]]))

    def test_2d_right_fixed(self):
        def dir_domain(x):
            dir01 = (x[0] == 1)
            return dir01

        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_2d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[0],
                                        [1],
                                        [0],
                                        [1],
                                        [0]]))

    def test_2d_bottom_fixed(self):
        def dir_domain(x):
            dir10 = (x[1] == 0)
            return dir10

        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_2d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[1, 0, 0, 0, 0],
                                                  [0, 1, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [1],
                                        [0],
                                        [0],
                                        [0]]))

    def test_2d_top_fixed(self):
        def dir_domain(x):
            dir11 = (x[1] == 1)
            return dir11

        dir_bc = pysofe.pde.conditions.DirichletBC(fe_space=fes_2d,
                                                   domain=dir_domain,
                                                   g=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = dir_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 1, 0, 0],
                                                  [0, 0, 0, 1, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[0],
                                        [0],
                                        [1],
                                        [1],
                                        [0]]))


class TestNeumannBC(object):

    def test_1d_both_ends(self):
        def neu_domain(x):
            neu0 = np.logical_or(x[0] == 0, x[0] == 1)
            return neu0
        
        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_1d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((4,4))
        b = np.zeros((4,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]))
        assert np.allclose(b, np.array([[1],
                                        [0],
                                        [0],
                                        [1]]))

    def test_1d_start(self):
        def neu_domain(x):
            neu0 = (x[0] == 0)
            return neu0
        
        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_1d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((4,4))
        b = np.zeros((4,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]))
        assert np.allclose(b, np.array([[1],
                                        [0],
                                        [0],
                                        [0]]))

    def test_1d_end(self):
        def neu_domain(x):
            neu0 = (x[0] == 1)
            return neu0
        
        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_1d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((4,4))
        b = np.zeros((4,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0],
                                                  [0, 0, 0, 0]]))
        assert np.allclose(b, np.array([[0],
                                        [0],
                                        [0],
                                        [1]]))

    def test_2d_all_sides(self):
        def neu_domain(x):
            neu0 = np.logical_or(x[0] == 0., x[0] == 1.)
            neu1 = np.logical_or(x[1] == 0., x[1] == 1.)
            return np.logical_or(neu0, neu1)

        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_2d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [1],
                                        [1],
                                        [1],
                                        [0]]))

    def test_2d_left_right(self):
        def neu_domain(x):
            neu0 = np.logical_or(x[0] == 0., x[0] == 1.)
            return neu0

        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_2d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [1],
                                        [1],
                                        [1],
                                        [0]]))

    def test_2d_top_bottom(self):
        def neu_domain(x):
            neu1 = np.logical_or(x[1] == 0., x[1] == 1.)
            return neu1

        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_2d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [1],
                                        [1],
                                        [1],
                                        [0]]))

    def test_2d_left(self):
        def neu_domain(x):
            neu00 = (x[0] == 0)
            return neu00

        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_2d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [0],
                                        [1],
                                        [0],
                                        [0]]))

    def test_2d_right(self):
        def neu_domain(x):
            neu01 = (x[0] == 1)
            return neu01

        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_2d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[0],
                                        [1],
                                        [0],
                                        [1],
                                        [0]]))

    def test_2d_bottom(self):
        def neu_domain(x):
            neu10 = (x[1] == 0)
            return neu10

        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_2d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[1],
                                        [1],
                                        [0],
                                        [0],
                                        [0]]))

    def test_2d_top(self):
        def neu_domain(x):
            neu11 = (x[1] == 1)
            return neu11

        neu_bc = pysofe.pde.conditions.NeumannBC(fe_space=fes_2d,
                                                 domain=neu_domain,
                                                 h=1)

        A = sparse.lil_matrix((5,5))
        b = np.zeros((5,1))

        A, b = neu_bc.apply(A, b)

        assert np.allclose(A.toarray(), np.array([[0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0],
                                                  [0, 0, 0, 0, 0,]]))
        assert np.allclose(b, np.array([[0],
                                        [0],
                                        [1],
                                        [1],
                                        [0]]))
