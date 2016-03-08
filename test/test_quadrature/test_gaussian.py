"""
Tests the Gaussian quadrature rules.
"""

import numpy as np
import pytest

from pysofe_light.quadrature.gaussian import GaussQuadSimp

# set global shortcut and tolerance
fac = np.math.factorial
eps = 1e-8

class TestGaussQuadSimp(object):
    order = 2
    quad_rule = GaussQuadSimp(order, dimension=3)

    def test_specs(self):
        assert len(self.quad_rule.points) == 4
        assert len(self.quad_rule.weights) == 4

        assert all([p.ndim == 2 for p in self.quad_rule.points])
        assert all([w.ndim == 1 for w in self.quad_rule.weights])

    def test_monomials_2D(self):
        x0, x1 = self.quad_rule.points[2]
        w = self.quad_rule.weights[2]

        for p in xrange(self.quad_rule.order + 1):
            for q in xrange(self.quad_rule.order - p + 1):
                quad = (np.power(x0, p) * np.power(x1, q) * w).sum()
                exact = (fac(p) * fac(q)) / float(fac(2 + p + q))

                assert abs(quad - exact) < eps

    def test_monomials_3D(self):
        x0, x1, x2 = self.quad_rule.points[3]
        w = self.quad_rule.weights[3]

        for p in xrange(self.quad_rule.order + 1):
            for q in xrange(self.quad_rule.order - p + 1):
                for r in xrange(self.quad_rule.order - p - q + 1):
                    quad = (np.power(x0, p) * np.power(x1, q) * np.power(x2, r) * w).sum()
                    exact = (fac(p) * fac(q) * fac(r)) / float(fac(3 + p + q + r))

                    assert abs(quad - exact) < eps
