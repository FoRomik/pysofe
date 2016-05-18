"""
Tests the mesh refinement routines.
"""

import numpy as np
import pytest

from pysofe.meshes.mesh import Mesh

class TestRefinement1D(object):
    mesh = Mesh(nodes=np.array([[0.],
                                [0.5],
                                [1.]]),
                connectivity=np.array([[1, 2],
                                [2, 3]]))

    def test_uniform_refine(self):
        self.mesh.refine(method='uniform', times=1)

        assert np.allclose(self.mesh.nodes,
                           np.array([[0.],
                                     [0.25],
                                     [0.5],
                                     [0.75],
                                     [1.]]))

        assert np.allclose(self.mesh.cells,
                           np.array([[1, 2],
                                     [2, 3],
                                     [3, 4],
                                     [4, 5]]))

class TestRefinement2D(object):
    mesh = Mesh(nodes=np.array([[ 0. ,  0. ],
                                [ 1. ,  0. ],
                                [ 0. ,  1. ],
                                [ 1. ,  1. ]]),
                connectivity=np.array([[1, 2, 3],
                                [2, 3, 4]]))


    def test_uniform_refine(self):
        self.mesh.refine(method='uniform', times=1)

        assert np.allclose(self.mesh.nodes,
                           np.array([[ 0. ,  0. ],
                                     [ 1. ,  0. ],
                                     [ 0. ,  1. ],
                                     [ 1. ,  1. ],
                                     [ 0.5,  0. ],
                                     [ 0. ,  0.5],
                                     [ 0.5,  0.5],
                                     [ 1. ,  0.5],
                                     [ 0.5,  1. ]]))

        assert np.allclose(self.mesh.cells,
                           np.array([[1, 5, 6],
                                     [2, 7, 8],
                                     [2, 5, 7],
                                     [3, 7, 9],
                                     [3, 6, 7],
                                     [4, 8, 9],
                                     [5, 6, 7],
                                     [7, 8, 9]]))

class TestRefinement3D(object):
    mesh = Mesh(nodes=np.array([[0., 0., 0.],
                                [1., 0., 0.],
                                [0., 1., 0.],
                                [0., 0., 1.],
                                [1., 1., 1.]]),
                connectivity=np.array([[1, 2, 3, 4],
                                       [2, 3, 4, 5]]))

    def test_uniform_refine(self):
        self.mesh.refine(method='uniform', times=1)

        assert np.allclose(self.mesh.nodes,
                           np.array([[0., 0., 0.],
                                     [1., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 1.],
                                     [1., 1., 1.],
                                     [.5, 0., 0.],
                                     [0., .5, 0.],
                                     [0., 0., .5],
                                     [.5, .5, 0.],
                                     [.5, 0., .5],
                                     [1., .5, .5],
                                     [0., .5, .5],
                                     [.5, 1., .5],
                                     [.5, .5, 1.]]))

        assert np.allclose(self.mesh.cells,
                           np.array([[1, 6, 7, 8],
                                     [2, 9, 10, 11],
                                     [2, 6, 9, 10],
                                     [3, 9, 12, 13],
                                     [3, 7, 9, 12],
                                     [4, 10, 12, 14],
                                     [4, 8, 10, 12],
                                     [5, 11, 13, 14],
                                     [6, 7, 8, 12],
                                     [9, 10, 11, 14],
                                     [6, 7, 9, 12],
                                     [9, 10, 12, 14],
                                     [6, 8, 10, 12],
                                     [9, 11, 13, 14],
                                     [6, 9, 10, 12],
                                     [9, 12, 13, 14]]))
