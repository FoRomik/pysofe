"""
Tests the operator data structures.
"""

import numpy as np
import pytest

from pysofe.meshes.mesh import Mesh
from pysofe.elements.simple.lagrange import P1
from pysofe.spaces.space import FESpace
from pysofe.spaces import operators

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

mesh_2d = Mesh(nodes_2d, cells_2d)

# create reference element and fe space
element_2d = P1(dimension=2)
fes_2d = FESpace(mesh_2d, element_2d)

class TestMassMatrix(object):
    pass
