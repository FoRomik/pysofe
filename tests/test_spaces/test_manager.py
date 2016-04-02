"""
Tests the dof manager data structure.
"""

import numpy as np
import pytest

from pysofe.meshes import Mesh
from pysofe.elements.base import Element
from pysofe.elements import P1
from pysofe.spaces.manager import DOFManager

# create test meshes and elements

# 1D
#----
nodes_1d = np.array([[0.], [.3], [.6], [1.]])
cells_1d = np.array([[1,2], [2,3], [3,4]])
mesh_1d = Mesh(nodes_1d, cells_1d)

elem_1d = Element(dimension=1, order=3, n_basis=(1,4), n_verts=(1,2))
elem_1d._dof_tuple = (1, 2)

# 2D
#----
nodes_2d = np.array([[0.,0.], [1.,0.], [0.,1.], [1.,1.]])
cells_2d = np.array([[1,2,3], [2,3,4]])
mesh_2d = Mesh(nodes_2d, cells_2d)

elem_2d = Element(dimension=2, order=3, n_basis=(1,4,10), n_verts=(1,2,3))
elem_2d._dof_tuple = (1, 2, 1)

# 3D
#----
nodes_3d = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [1.,1.,1.]])
cells_3d = np.array([[1, 2, 3, 4], [2, 3, 4, 5]])
mesh_3d = Mesh(nodes_3d, cells_3d)

elem_3d = Element(dimension=3, order=4, n_basis=(1,5,15,35), n_verts=(1,2,3,4))
elem_3d._dof_tuple = (1, 3, 3, 1)

class TestDOFManager1DP3(object):
    dm = DOFManager(mesh_1d, elem_1d)

    def test_n_dof(self):
        assert self.dm.n_dof == 10

    def test_dof_map_vertices(self):
        dof_map = self.dm.get_dof_map(d=0)

        assert np.all(dof_map == np.array([[1, 2, 3, 4]]))

    def test_dof_map_cells(self):
        dof_map = self.dm.get_dof_map(d=1)

        assert dof_map.shape[0] == self.dm._element.n_basis[1]
        assert np.all(dof_map == np.array([[1, 2, 3],
                                           [2, 3, 4],
                                           [5, 6, 7],
                                           [8, 9, 10]]))

    def test_extract_dofs_vertices(self):
        dofs = self.dm.extract_dofs(d=0)

        assert np.allclose(dofs, np.array([True,  True,  True,  True, False,
                                           False, False, False, False, False]))

    def test_extract_dofs_edges(self):
        dofs = self.dm.extract_dofs(d=1)

        assert np.allclose(dofs, np.array([True, True, True, True, True,
                                           True, True, True, True, True]))

class TestDOFManager2DP3(object):
    dm = DOFManager(mesh_2d, elem_2d)

    def test_n_dof(self):
        assert self.dm.n_dof == 16

    def test_dof_map_vertices(self):
        dof_map = self.dm.get_dof_map(d=0)

        assert np.all(dof_map == np.array([[1, 2, 3, 4]]))

    def test_dof_map_edges(self):
        dof_map = self.dm.get_dof_map(d=1)

        assert dof_map.shape[0] == self.dm._element.n_basis[1]
        assert np.all(dof_map == np.array([[ 1,  1,  2,  2,  3],
                                           [ 2,  3,  3,  4,  4],
                                           [ 5,  6,  7,  8,  9],
                                           [10, 11, 12, 13, 14]]))

    def test_dof_map_cells(self):
        dof_map = self.dm.get_dof_map(d=2)

        assert dof_map.shape[0] == self.dm._element.n_basis[2]
        assert np.all(dof_map == np.array([[ 1,  2],
                                           [ 2,  3],
                                           [ 3,  4],
                                           [ 5,  7],
                                           [ 6,  8],
                                           [ 7,  9],
                                           [10, 12],
                                           [11, 13],
                                           [12, 14],
                                           [15, 16]]))

    def test_extract_dofs_vertices(self):
        dofs = self.dm.extract_dofs(d=0)

        assert np.allclose(dofs, np.array([True,  True,  True,  True,
                                           False, False, False, False,
                                           False, False, False, False,
                                           False, False, False, False]))

    def test_extract_dofs_edges(self):
        dofs = self.dm.extract_dofs(d=1)

        assert np.allclose(dofs, np.array([True, True, True, True,
                                           True, True, True, True,
                                           True, True, True, True,
                                           True, True, False, False]))

    def test_extract_dofs_cells(self):
        dofs = self.dm.extract_dofs(d=2)

        assert np.allclose(dofs, np.array([True, True, True, True,
                                           True, True, True, True,
                                           True, True, True, True,
                                           True, True, True, True]))

class TestDOFManager3DP4(object):
    dm = DOFManager(mesh_3d, elem_3d)

    def test_n_dof(self):
        assert self.dm.n_dof == 55

    def test_dof_map_vertices(self):
        dof_map = self.dm.get_dof_map(d=0)

        assert np.all(dof_map == np.array([[1, 2, 3, 4, 5]]))

    def test_dof_map_edges(self):
        dof_map = self.dm.get_dof_map(d=1)

        assert dof_map.shape[0] == self.dm._element.n_basis[1]
        assert np.all(dof_map == np.array([[ 1,  1,  1,  2,  2,  2,  3,  3,  4],
                                           [ 2,  3,  4,  3,  4,  5,  4,  5,  5],
                                           [ 6,  7,  8,  9, 10, 11, 12, 13, 14],
                                           [15, 16, 17, 18, 19, 20, 21, 22, 23],
                                           [24, 25, 26, 27, 28, 29, 30, 31, 32]]))

    def test_dof_map_faces(self):
        dof_map = self.dm.get_dof_map(d=2)

        assert dof_map.shape[0] == self.dm._element.n_basis[2]
        assert np.all(dof_map == np.array([[ 1,  1,  1,  2,  2,  2,  3],
                                           [ 2,  2,  3,  3,  3,  4,  4],
                                           [ 3,  4,  4,  4,  5,  5,  5],
                                           [ 6,  6,  7,  9,  9, 10, 12],
                                           [ 7,  8,  8, 10, 11, 11, 13],
                                           [ 9, 10, 12, 12, 13, 14, 14],
                                           [15, 15, 16, 18, 18, 19, 21],
                                           [16, 17, 17, 19, 20, 20, 22],
                                           [18, 19, 21, 21, 22, 23, 23],
                                           [24, 24, 25, 27, 27, 28, 30],
                                           [25, 26, 26, 28, 29, 29, 31],
                                           [27, 28, 30, 30, 31, 32, 32],
                                           [33, 34, 35, 36, 37, 38, 39],
                                           [40, 41, 42, 43, 44, 45, 46],
                                           [47, 48, 49, 50, 51, 52, 53]]))

    def test_dof_map_cells(self):
        dof_map = self.dm.get_dof_map(d=3)

        assert dof_map.shape[0] == self.dm._element.n_basis[3]
        assert np.all(dof_map == np.array([[ 1,  2],
                                           [ 2,  3],
                                           [ 3,  4],
                                           [ 4,  5],
                                           [ 6,  9],
                                           [ 7, 10],
                                           [ 8, 11],
                                           [ 9, 12],
                                           [10, 13],
                                           [12, 14],
                                           [15, 18],
                                           [16, 19],
                                           [17, 20],
                                           [18, 21],
                                           [19, 22],
                                           [21, 23],
                                           [24, 27],
                                           [25, 28],
                                           [26, 29],
                                           [27, 30],
                                           [28, 31],
                                           [30, 32],
                                           [33, 36],
                                           [34, 37],
                                           [35, 38],
                                           [36, 39],
                                           [40, 43],
                                           [41, 44],
                                           [42, 45],
                                           [43, 46],
                                           [47, 50],
                                           [48, 51],
                                           [49, 52],
                                           [50, 53],
                                           [54, 55]]))
