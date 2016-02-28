"""
Tests for the families of reference maps.
"""

import numpy as np
import pytest

from pysofe_light import meshes

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

# coordinates of reference domain vertices
verts_1d = np.array([[0., 1.]])

verts_2d = np.array([[0., 1., 0.],
                     [0., 0., 1.]])

verts_3d = np.array([[0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.]])


class TestReferenceMap2D(object):
    mesh = meshes.Mesh(nodes_2d, cells_2d)
    ref_map = meshes.reference_map.ReferenceMap(mesh=mesh)
    
    def test_d0_evaluation_to_elems(self):
        # evaluating ref map with reference domain
        # 2d vertices should give mesh cell nodes
        RM = self.ref_map.eval(points=verts_2d, d=0)

        nE = self.mesh.cells.shape[0]
        nD = self.mesh.dimension
        nP = verts_2d.shape[1]
        
        assert RM.shape == (nE, nP, nD)
        assert np.allclose(RM, self.mesh.nodes.take(self.mesh.cells - 1, axis=0))

    def test_d0_evaluation_to_edges(self):
        # evaluating ref map with reference domain
        # 1d vertices should give mesh edges nodes
        RM = self.ref_map.eval(points=verts_1d, d=0)

        nE = self.mesh.edges.shape[0]
        nD = self.mesh.dimension
        nP = verts_1d.shape[1]
        
        assert RM.shape == (nE, nP, nD)
        assert np.allclose(RM, self.mesh.nodes.take(self.mesh.edges - 1, axis=0))

    def test_d1_evaluation_to_cells(self):
        # evaluating ref map 1st derivative to cells should give matrix
        # that has scaled basis vectors of cell coordinate systems as columns
        dRM = self.ref_map.eval(points=verts_2d, d=1)

        nE = self.mesh.cells.shape[0]
        nD = self.mesh.dimension
        nP = verts_2d.shape[1]

        E1 = [[1,0],[0,1]];  E3 = [[-1,-1],[0,1]]; E5 = [[0,1],[-1,-1]]; E7 = [[-1,0],[1,1]]
        E2 = [[-1,0],[1,1]]; E4 = [[1,1],[-1,0]];  E6 = [[0,-1],[-1,0]]; E8 = [[1,0],[0,1]]
        E = 0.5 * np.array([E1, E2, E3, E4, E5, E6, E7, E8])
        
        assert dRM.shape == (nE, nP, nD, nD)
        assert np.allclose(dRM, np.tile(E[:,None], (nP,1,1)))

    def test_d1_evaluation_to_edges(self):
        # evaluating ref map 1st derivative to edges should give vectors
        # that point in direction of each edge,
        # i.e. `basis vector` for each edge
        dRM = self.ref_map.eval(points=verts_1d, d=1)

        nE = self.mesh.edges.shape[0]
        nD = self.mesh.dimension
        nP = verts_1d.shape[1]

        E = 0.5 * np.array([[1,0], [0,1], [-1,0], [-1,1], [0,1], [0,-1],
                            [1,-1], [1,0], [0,-1], [-1,0], [-1,1], [0,1],
                            [1,0], [1,0], [0,1], [-1,1]])
        
        assert dRM.shape == (nE, nP, nD, 1)
        assert np.allclose(dRM[...,0], np.tile(E[:,None], (nP,1)))

    def test_jacobian_inverse_cells(self):
        jacs = self.ref_map.eval(points=verts_2d, d=1)
        jacs_inv = self.ref_map.jacobian_inverse(points=verts_2d)

        eyes = (jacs[:,:,:,:,None] * jacs_inv[:,:,None,:,:]).sum(axis=-2)
        eye = np.eye(2)

        nE = self.mesh.cells.shape[0]
        nP = verts_2d.shape[1]
        
        assert jacs.shape == jacs_inv.shape
        assert np.allclose(eyes, np.tile(eye, (nE,nP,1,1)))

    def test_jacobian_determinat_cells(self):
        # the jacobian determinant absoulut value should be equal to
        # two times the area of the cells
        jacs_det = self.ref_map.jacobian_determinant(points=verts_2d)

        nE = self.mesh.cells.shape[0]
        nP = verts_2d.shape[1]

        # all cells have same area
        area = 0.125

        assert np.allclose(np.abs(jacs_det), 2 * area * np.ones((nE,nP)))

        
