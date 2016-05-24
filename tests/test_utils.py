"""
Tests the utility functions.
"""

import numpy as np
import pytest

from pysofe import utils

def test_match_nodes():
    n_nodes = 1000
    n_dims = 3

    for D in xrange(2, n_dims):
        dim = np.random.randint(0, D, size=1)
        
        nodes0 = np.random.random(size=(n_nodes, D))
        nodes1 = nodes0.copy(); nodes1[:,dim] = 0
        
        I, J = utils.match_nodes(nodes0, nodes1, dim)
        
        assert all(np.allclose(nodes0[I,d], nodes1[J,d]) for d in xrange(2) if d != dim)
