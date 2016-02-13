"""
Tests for mesh topologies.
"""

import numpy as np
import pytest

from pysofe_light import meshes

# the 1D test mesh
#
# 1---(1)---2---(2)---3---(3)---4

class TestMeshTopology1D(object):
    # the 1D test mesh connectivity array
    cells1D  = np.array([[1,2],
                         [2,3],
                         [3,4]])

    topo = meshes.topology.MeshTopology(cells=cells1D, dimension=1)

    def test_attributes(self):
        assert self.topo._dimension == 1
        assert np.all(self.topo._n_vertices == [1, 2])

    def test_incidence_1_0_and_0_1(self):
        assert np.all(self.topo.get_connectivity(1,0).toarray()
                      == np.array([[1,1,0,0],
                                   [0,1,1,0],
                                   [0,0,1,1]]))

        assert np.all(self.topo.get_connectivity(0,1).toarray()
                      == np.array([[1,0,0],
                                   [1,1,0],
                                   [0,1,1],
                                   [0,0,1]]))

# the 2D test mesh
#
# 4---------3
# |\       /|
# | \ (3) / |
# |  \   /  |
# |   \ /   |
# |(4) 5 (2)|
# |   / \   |
# |  /   \  |
# | / (1) \ |
# |/       \|
# 1---------2

class TestMeshTopology2D(object):
    # the 2D test mesh connectivity array
    cells2D = np.array([[1,2,5],
                        [2,3,5],
                        [3,4,5],
                        [4,1,5]])

    topo = meshes.topology.MeshTopology(cells=cells2D, dimension=2)

    def test_attributes(self):
        assert self.topo._dimension == 2
        assert np.all(self.topo._n_vertices == [1, 2, 3])

    def test_incidence_2_0_and_0_2(self):
        assert np.all(self.topo.get_connectivity(2,0).toarray()
                      == np.array([[1,1,0,0,1],
                                   [0,1,1,0,1],
                                   [0,0,1,1,1],
                                   [1,0,0,1,1]]))

        assert np.all(self.topo.get_connectivity(0,2).toarray()
                      == np.array([[1,0,0,1],
                                   [1,1,0,0],
                                   [0,1,1,0],
                                   [0,0,1,1],
                                   [1,1,1,1]]))

    def test_incidence_1_0_and_0_1(self):
        assert np.all(self.topo.get_connectivity(1,0).toarray()
                      == np.array([[1,1,0,0,0],
                                   [1,0,0,1,0],
                                   [1,0,0,0,1],
                                   [0,1,1,0,0],
                                   [0,1,0,0,1],
                                   [0,0,1,1,0],
                                   [0,0,1,0,1],
                                   [0,0,0,1,1]]))

        assert np.all(self.topo.get_connectivity(0,1).toarray()
                      == np.array([[1,1,1,0,0,0,0,0],
                                   [1,0,0,1,1,0,0,0],
                                   [0,0,0,1,0,1,1,0],
                                   [0,1,0,0,0,1,0,1],
                                   [0,0,1,0,1,0,1,1]]))

    def test_incidence_2_1(self):
        assert np.all(self.topo.get_connectivity(2,1).toarray()
                      == np.array([[1,0,1,0,1,0,0,0],
                                   [0,0,0,1,1,0,1,0],
                                   [0,0,0,0,0,1,1,1],
                                   [0,1,1,0,0,0,0,1]]))

        
class TestMeshTopology3D(object):
    # the 3D test mesh connectivity array
    cells3D = np.array([[1,2,3,5],
                        [3,4,1,5]])

    topo = meshes.topology.MeshTopology(cells=cells3D, dimension=3)

    def test_attributes(self):
        assert self.topo._dimension == 3
        assert np.all(self.topo._n_vertices == [1, 2, 3, 4])

    def test_incidence_3_0_and_0_3(self):
        assert np.all(self.topo.get_connectivity(3,0).toarray()
                      == np.array([[1,1,1,0,1],
                                   [1,0,1,1,1]]))

        assert np.all(self.topo.get_connectivity(0,3).toarray()
                      == np.array([[1,1],
                                   [1,0],
                                   [1,1],
                                   [0,1],
                                   [1,1]]))

    def test_incidence_2_0_and_0_2(self):
        assert np.all(self.topo.get_connectivity(2,0).toarray()
                      == np.array([[1,1,1,0,0],
                                   [1,1,0,0,1],
                                   [1,0,1,1,0],
                                   [1,0,1,0,1],
                                   [1,0,0,1,1],
                                   [0,1,1,0,1],
                                   [0,0,1,1,1]]))

        assert np.all(self.topo.get_connectivity(0,2).toarray()
                      == np.array([[1,1,1,1,1,0,0],
                                   [1,1,0,0,0,1,0],
                                   [1,0,1,1,0,1,1],
                                   [0,0,1,0,1,0,1],
                                   [0,1,0,1,1,1,1]]))

    def test_incidence_1_0_and_0_1(self):
        assert np.all(self.topo.get_connectivity(1,0).toarray()
                      == np.array([[1,1,0,0,0],
                                   [1,0,1,0,0],
                                   [1,0,0,1,0],
                                   [1,0,0,0,1],
                                   [0,1,1,0,0],
                                   [0,1,0,0,1],
                                   [0,0,1,1,0],
                                   [0,0,1,0,1],
                                   [0,0,0,1,1]]))

        assert np.all(self.topo.get_connectivity(0,1).toarray()
                      == np.array([[1,1,1,1,0,0,0,0,0],
                                   [1,0,0,0,1,1,0,0,0],
                                   [0,1,0,0,1,0,1,1,0],
                                   [0,0,1,0,0,0,1,0,1],
                                   [0,0,0,1,0,1,0,1,1]]))

if __name__ == '__main__':
    from IPython import embed as IPS
    IPS()
