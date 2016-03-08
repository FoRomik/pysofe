"""
Tests for simple finite elements.
"""

import numpy as np
import pytest

from pysofe_light import elements

simplicial_vertices = dict()
simplicial_vertices[1] = np.array([[0., 1.]])
simplicial_vertices[2] = np.array([[0., 1., 0.],
                                   [0., 0., 1.]])
simplicial_vertices[3] = np.array([[0., 1., 0., 0.],
                                   [0., 0., 1., 0.],
                                   [0., 0., 0., 1.]])

class TestLagrangeP1(object):
    elem = elements.simple.lagrange.P1(dimension=3)
    
    def test_specs(self):
        assert self.elem.dimension == 3
        assert self.elem.n_verts == (2, 3, 4)
        assert self.elem.order == 1
        assert self.elem.n_basis == (2, 3, 4)
        assert self.elem.dof_tuple == (1, 0, 0, 0)

    def test_eval_d0basis(self):
        nerr = 0
        failed = []
        
        for dim in xrange(1, self.elem.dimension + 1):
            basis = self.elem.eval_basis(points=simplicial_vertices[dim], deriv=0)

            try:
                assert basis.shape == (dim+1, dim+1)
                assert np.allclose(basis, np.eye(dim+1))
            except AssertionError:
                failed.append(dim)
                nerr += 1

        if failed:
            msg = '{} D0 evaluation failed, dimensions: ({})'.format(nerr, failed)
            pytest.fail(msg)

    def test_eval_d1basis(self):
        nerr = 0
        failed = []
        
        for dim in xrange(1, self.elem.dimension + 1):
            dbasis = self.elem.eval_basis(points=simplicial_vertices[dim], deriv=1)
            nV = dim + 1

            try:
                assert dbasis.shape == (dim+1, nV, dim)

                I = np.ones((nV, 1))
                O = np.zeros((nV, 1))

                if dim == 1:
                    assert np.allclose(dbasis, np.array([np.c_[-I],
                                                        np.c_[ I]]))
                elif dim == 2:
                    assert np.allclose(dbasis, np.array([np.c_[-I, -I],
                                                        np.c_[ I,  O],
                                                        np.c_[ O,  I]]))
                elif dim == 3:
                    assert np.allclose(dbasis, np.array([np.c_[-I, -I, -I],
                                                        np.c_[ I,  O,  O],
                                                        np.c_[ O,  I,  O],
                                                        np.c_[ O,  O,  I]]))
                else:
                    raise ValueError('Invalid dimension ({})'.format(dim))
                
            except AssertionError:
                failed.append(dim)
                nerr += 1

        if failed:
            msg = '{} D1 evaluation failed, dimensions: ({})'.format(nerr, failed)
            pytest.fail(msg)

    def test_eval_d2basis(self):
        nerr = 0
        failed = []
        
        for dim in xrange(1, self.elem.dimension + 1):
            ddbasis = self.elem.eval_basis(points=simplicial_vertices[dim], deriv=2)
            nV = dim + 1
            
            try:
                assert ddbasis.shape == (dim+1, nV, dim, dim)
                assert np.allclose(ddbasis, np.zeros((dim+1, nV, dim, dim)))
            except AssertionError:
                failed.append(dim)
                nerr += 1

        if failed:
            msg = '{} D2 evaluation failed, dimensions: ({})'.format(nerr, failed)
            pytest.fail(msg)
