"""
Provides the data structure for numerical integration.

The purpose of these data structures is to give access
to the quadrature points and weights for several spatial domains.
"""

from .simplicial import GaussPoint, GaussInterval, GaussTriangle, GaussTetrahedron
