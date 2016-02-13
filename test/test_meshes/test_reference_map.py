"""
Tests for the families of reference maps.
"""

import numpy as np
import pytest

from pysofe_light import elements, meshes

class TestReferenceMap2D(object):
    nodes = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    cells = np.array([[1, 2, 3], [2, 3, 4]])
    mesh = meshes.Mesh(nodes, cells)

    ref_map = meshes.reference_map.ReferenceMap(mesh=mesh)
    
    elem_vertices = np.array([[0., 1., 0.],
                              [0., 0., 1.]])
    edge_vertices = np.array([[0., 1.]])

    def test_d0_evaluation_to_elems(self):
        RM = self.ref_map.eval(points=self.elem_vertices, d=0)

        nE = self.mesh.cells.shape[0]
        nP = 3
        nD = 2
        
        assert RM.shape == (nE, nP, nD)
        assert np.allclose(RM, self.mesh.nodes.take(self.mesh.cells - 1, axis=0))

    def test_d0_evaluation_to_edges(self):
        RM = self.ref_map.eval(points=self.edge_vertices, d=0)

        nE = self.mesh.edges.shape[0]
        nP = 2
        nD = 2
        
        assert RM.shape == (nE, nP, nD)
        assert np.allclose(RM, self.mesh.nodes.take(self.mesh.edges - 1, axis=0))

    def test_d1_evaluation_to_elems(self):
        dRM = self.ref_map.eval(points=self.elem_vertices, d=1)

        nE = self.mesh.cells.shape[0]
        nP = 3
        nD = 2

        E1 = np.array([[1.,0.],
                       [0.,1.]])

        E2 = np.array([[-1.,0.],
                       [ 1.,1.]])
        
        assert dRM.shape == (nE, nP, nD, nD)
        assert np.allclose(dRM, np.array([[E1]*3, [E2]*3]))

    def test_d1_evaluation_to_edges(self):
        dRM = self.ref_map.eval(points=self.edge_vertices, d=1)

        nE = self.mesh.edges.shape[0]
        nP = 2
        nD = 2

        e1, e2, e3, e4, e5 = ([[-1], [1]] * self.mesh.nodes.take(self.mesh.edges-1, axis=0)).sum(axis=1)

        assert dRM.shape == (nE, nP, nD, 1)
        assert np.allclose(dRM, np.array([[e1]*nP, [e2]*nP, [e3]*nP, [e4]*nP, [e5]*nP])[:,:,:,None])

