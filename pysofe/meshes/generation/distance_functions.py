"""
Provides some signed distance functions for the implicit
definition of a mesh geometry.
"""

# IMPORTS
import numpy as np

class SignedDistanceFunction(object):
    """
    Base class for all signed distance functions.
    """

    def __init__(self, bbox=None):
        self._bbox = bbox

    def __call__(self, points):
        return self._evaluate(points)

    def _evaluate(self, points):
        raise NotImplementedError()

    @property
    def bbox(self):
        """
        A bounding box of the geometry specified by this distance function.
        """
        return self._bbox

class DUnion(SignedDistanceFunction):
    """
    Signed distance function for the union of several geometries.
    """

    def __init__(self, *args):
        assert all(callable(arg) for arg in args)

        # determine overall bounding box
        # (only possible if all args belong to sdf class)
        if all(isinstance(arg, SignedDistanceFunction) for arg in args):
            B = np.array([sdf.bbox for sdf in args])

            b_min = B[:,:,0].min(axis=0)
            b_max = B[:,:,1].max(axis=0)

            box = np.vstack([b_min, b_max]).T
        else:
            box = None

        SignedDistanceFunction.__init__(self, bbox=box)

        # save sdf list
        self.geometries = args

    def _evaluate(self, points):
        return np.min([sdf(points) for sdf in self.geometries], axis=0)

class DIntersection(SignedDistanceFunction):
    """
    Signed distance function for the intersection of several geometries.
    """

    def __init__(self, *args):
        assert all(callable(arg) for arg in args)

        # determine overall bounding box
        # (only possible if all args belong to sdf class)
        if all(isinstance(arg, SignedDistanceFunction) for arg in args):
            B = np.array([sdf.bbox for sdf in args])

            b_min = B[:,:,0].max(axis=0)
            b_max = B[:,:,1].min(axis=0)

            box = np.vstack([b_min, b_max]).T
        else:
            box = None

        SignedDistanceFunction.__init__(self, bbox=box)

        # save sdf list
        self.geometries = args

    def _evaluate(self, points):
        return np.max([sdf(points) for sdf in self.geometries], axis=0)

class DDifference(SignedDistanceFunction):
    """
    Signed distance function for the difference of two geometries.
    """

    def __init__(self, arg0, arg1):
        assert all(callable(arg) for arg in [arg0, arg1])

        # determine bounding box
        # (only possible if all arg0 belongs to sdf class)
        if isinstance(arg0, SignedDistanceFunction):
            box = arg0.bbox
        else:
            box = None

        SignedDistanceFunction.__init__(self, bbox=box)

        # save sdf list
        self.sdf0 = arg0
        self.sdf1 = arg1

    def _evaluate(self, points):
        return np.maximum(self.sdf0(points), -self.sdf1(points))

class DSphere(SignedDistanceFunction):
    """
    Signed distance function for n-dimensional sphere.

    Parameters
    ----------

    centre : array_like
        The centre of the sphere

    radius : float
        The radius of the sphere
    """

    def __init__(self, centre, radius):
        if not isinstance(centre, np.ndarray):
            centre = np.asarray(centre)

        assert centre.ndim == 1
        assert radius > 0

        # determine bounding box
        box = np.array([(c-radius, c+radius) for c in centre])

        SignedDistanceFunction.__init__(self, bbox=box)

        # save attributes
        self.centre = centre
        self.radius = radius

    def _evaluate(self, points):
        centre = self.centre[:,None]
        return np.sqrt(np.power(points - centre, 2).sum(axis=0)) - self.radius
    
class DOrthotope(SignedDistanceFunction):
    """
    Signed distance function for n-orthotope 
    (or n-dimensional box).

    Parameters
    ----------

    limits : array_like
        The edge limits in each direction
    """

    def __init__(self, limits):
        if not isinstance(limits, np.ndarray):
            limits = np.asarray(limits)

        assert limits.ndim == 2
        assert limits.shape[1] == 2 # min/max limit

        SignedDistanceFunction.__init__(self, bbox=limits)

        self.limits = limits

    def _evaluate(self, points):
        limits_min = self.limits[:,0,None]
        limits_max = self.limits[:,1,None]

        min_dists = np.minimum(points - limits_min,
                               limits_max - points)

        return -min_dists.min(axis=0)

class DCircle(DSphere):
    """
    Signed distance function for a circle.

    Parameters
    ----------

    centre : array_like
        The centre of the circle

    radius : float
        The radius of the circle
    """

    def __init__(self, centre=(0, 0), radius=1):
        assert np.size(centre) == 2

        DSphere.__init__(self, centre, radius)

class DRectangle(DOrthotope):
    """
    Signed distance function for a rectangle.
    
    Parameters
    ----------

    xlim : tuple
        Limits on x-axis ([xmin, xmax])

    ylim : tuple
        Limits on y-axis ([ymin, ymax])
    """

    def __init__(self, xlim=(0, 1), ylim=(0, 1)):
        lim = np.array([xlim,
                        ylim])
        
        DOrthotope.__init__(self, limits=lim)
