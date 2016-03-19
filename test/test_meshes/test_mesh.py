"""
Tests the mesh data structure.
"""

import numpy as np
import pytest

from pysofe.meshes.mesh import Mesh

# define mesh nodes and cells
nodes_2d = np.array([[ 0. ,  0. ],
                     [ 1. ,  0. ],
                     [ 0. ,  1. ],
                     [ 1. ,  1. ],
                     [ 0.5,  0. ],
                     [ 0. ,  0.5],
                     [ 0.5,  0.5],
                     [ 1. ,  0.5],
                     [ 0.5,  1. ]])

cells_2d = np.array([[1, 5, 6],
                     [2, 7, 8],
                     [2, 5, 7],
                     [3, 7, 9],
                     [3, 6, 7],
                     [4, 8, 9],
                     [5, 6, 7],
                     [7, 8, 9]])

# define some test functions
def scalar_fnc(x):
    return x[0] * x[1]

c0 = 1.

c1 = np.array([1., 1.])

c2 = np.array([[1., 1.],
               [1., 1.]])

# and some test points
local_points_2d = np.array([[0.,1.,0.,1/3.],
                            [0.,0.,1.,1/3.]])

global_points_2d = \
        np.array([[ 0.        ,  0.5       ,  0.        ,  0.16666667,  1.        ,
                    0.5       ,  1.        ,  0.83333333,  1.        ,  0.5       ,
                    0.5       ,  0.66666667,  0.        ,  0.5       ,  0.5       ,
                    0.33333333,  0.        ,  0.        ,  0.5       ,  0.16666667,
                    1.        ,  1.        ,  0.5       ,  0.83333333,  0.5       ,
                    0.        ,  0.5       ,  0.33333333,  0.5       ,  1.        ,
                    0.5       ,  0.66666667],
                  [ 0.        ,  0.        ,  0.5       ,  0.16666667,  0.        ,
                    0.5       ,  0.5       ,  0.33333333,  0.        ,  0.        ,
                    0.5       ,  0.16666667,  1.        ,  0.5       ,  1.        ,
                    0.83333333,  1.        ,  0.5       ,  0.5       ,  0.66666667,
                    1.        ,  0.5       ,  1.        ,  0.83333333,  0.        ,
                    0.5       ,  0.5       ,  0.33333333,  0.5       ,  0.5       ,
                    1.        ,  0.66666667]])

class TestMesh2D(object):
    mesh = Mesh(nodes_2d, cells_2d)

    def test_attributes(self):
        assert self.mesh.dimension == 2

    def test_mesh_nodes(self):
        assert np.all(self.mesh.nodes == nodes_2d)

    def test_mesh_edges(self):
        assert np.all(self.mesh.edges
                      == np.array([[1, 5],
                                   [1, 6],
                                   [2, 5],
                                   [2, 7],
                                   [2, 8],
                                   [3, 6],
                                   [3, 7],
                                   [3, 9],
                                   [4, 8],
                                   [4, 9],
                                   [5, 6],
                                   [5, 7],
                                   [6, 7],
                                   [7, 8],
                                   [7, 9],
                                   [8, 9]]))

    def test_mesh_cells(self):
        assert np.allclose(self.mesh.cells, cells_2d)

    def test_mesh_faces(self):
        assert np.allclose(self.mesh.faces, self.mesh.cells)

    def test_mesh_facets(self):
        assert np.allclose(self.mesh.facets, self.mesh.edges)

    def test_boundary(self):
        assert np.allclose(self.mesh.boundary(),
                           np.array([True, True, True, False,
                                     True, True, False, True,
                                     True, True, False, False,
                                     False, False, False, False]))

    def test_boundary_left_right(self):
        bnd_left_right = lambda x: np.logical_or(x[0] == 0., x[0] == 1.)
        
        assert np.allclose(self.mesh.boundary(fnc=bnd_left_right),
                           np.array([False, True, False, False,
                                     True, True, False, False,
                                     True, False, False, False,
                                     False, False, False, False]))

    def test_boundary(self):
        bnd_top_bottom = lambda x: np.logical_or(x[1] == 0., x[1] == 1.)
        
        assert np.allclose(self.mesh.boundary(fnc=bnd_top_bottom),
                           np.array([True, False, True, False,
                                     False, False, False, True,
                                     False, True, False, False,
                                     False, False, False, False]))

    def test_eval_function_callable_scalar(self):
        values = self.mesh.eval_function(scalar_fnc, local_points_2d)
        nE = self.mesh.cells.shape[0]
        nP = local_points_2d.shape[1]

        assert np.allclose(values,
                           global_points_2d.prod(axis=0).reshape((nE, nP)))

    def test_eval_function_constant_scalar(self):
        values = self.mesh.eval_function(c0, local_points_2d)
        nE = self.mesh.cells.shape[0]
        nP = local_points_2d.shape[1]
        
        assert np.allclose(values, np.ones((nE, nP)))

    def test_eval_function_constant_vector(self):
        values = self.mesh.eval_function(c1, local_points_2d)
        nE = self.mesh.cells.shape[0]
        nP = local_points_2d.shape[1]
        
        assert np.allclose(values, np.ones((nE, nP, 2)))

    def test_eval_function_constant_matrix(self):
        values = self.mesh.eval_function(c2, local_points_2d)
        nE = self.mesh.cells.shape[0]
        nP = local_points_2d.shape[1]
        
        assert np.allclose(values, np.ones((nE, nP, 2, 2)))
