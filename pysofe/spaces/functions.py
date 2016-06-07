"""
Provides convinience classes for functions in the fem framework.
"""

# IMPORTS
import numpy as np

# DEBUGGING
from IPython import embed as IPS

class FEFunction(object):
    """
    A finite element function defined via degrees of freedoms.

    Parameters
    ----------

    fe_space : pysofe.spaces.space.FESpace
        The function space

    dof_values : array_like
        Values for the degrees of freedom of the function space
    """

    def __init__(self, fe_space, dof_values):
        if not isinstance(dof_values, np.ndarray):
            dof_values = np.asarray(dof_values)

        assert dof_values.ndim == 1

        if not np.size(dof_values) == fe_space.n_dof:
            raise ValueError("fe space and dof values don't match!")
        
        self.fe_space = fe_space
        self.dofs = dof_values

    @property
    def order(self):
        """
        The polynomial order of the function.
        """
        return self.fe_space.element.order
        
    def __call__(self, points, deriv=0, local=False):
        return self._evaluate(points, deriv, local)
    
    def _evaluate(self, points, deriv=0, local=False):
        """
        Evaluates the function or its derivatives at given points.

        Parameters
        ----------

        points : array_like
            The local points at the reference domain

        deriv : int
            The derivation order
        """

        U = self.fe_space.eval_dofs(dofs=self.dofs, points=points,
                                    deriv=deriv, local=local)

        return U

class PeriodicFEFunction(FEFunction):
    """
    A periodic finite element function.

    Parameters
    ----------

    fe_space : pysofe.spaces.space.FESpace
        The function space

    dof_values : array_like
        Values for the degrees of freedom of the function space
    
    period : scalar, iterable
        Uniform period or period for each axis direction
    """

    def __init__(self, fe_space, dof_values, period):
        FEFunction.__init__(self, fe_space, dof_values)

        period = np.asarray(period).flatten()

        if period.size == 1 and fe_space.element.dimension > 1:
            period = np.repeat(period, fe_space.element.dimension)
        else:
            assert period.size == fe_space.element.dimension

        self._period = period

    @property
    def period(self):
        return self._period
        
    def _evaluate(self, points, deriv=0, local=True):
        # resolve periodicity if necessary
        if not local:
            points = np.mod(points, self.period[:,None])

        return super(PeriodicFEFunction, self)._evaluate(points, deriv, local)
        
class MeshFunction(object):
    """
    Wrapper for the evaluation of a given function on a specific mesh.

    Parameters
    ----------

    fnc : callable
        The function
    
    mesh : pysofe.meshes.mesh.Mesh
        The considered mesh
    """

    def __init__(self, fnc, mesh):
        self.fnc = fnc
        self.mesh = mesh

    def __call__(self, points, local=True):
        return self._evaluate(points, local)

    def _evaluate(self, points, local=True):
        """
        Evaluates the function at given points.
        """

        if not local:
            raise NotImplementedError()

        F = self.mesh.eval_function(fnc=self.fnc, points=points)

        return F
    
