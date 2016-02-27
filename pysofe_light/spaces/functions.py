"""
Provides convinience classes for functions in the fem framework.
"""

# IMPORTS
import numpy as np

class FEFunction(object):
    '''
    A finite element function defined via degrees of freedoms.

    Parameters
    ----------

    fe_space : pysofe.spaces.FESpace
        The function space

    dof_values : array_like
        Values for the degrees of freedom of the function space
    '''

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
        '''
        The polynomial order of the function.
        '''
        return self.fe_space.element.order
        
    def __call__(self, points, d=0, local=False):
        return self._evaluate(points, d, local)
    
    def _evaluate(self, points, d=0):
        '''
        Evaluates the function or its derivatives at given points.

        Parameters
        ----------

        points : array_like
            The local points at the reference domain

        d : int
            The derivation order
        '''

        # determine for which entities to evaluate the function
        dim = np.size(points, axis=0)

        # check input
        if dim < self.fe_space.mesh.dimension and d > 0:
            raise NotImplementedError('Higher order derivatives for traces not supported!')
        
        # get dof map and adjust values
        dof_map = self.fe_space._get_dof_map(d=dim)

        values = np.sign(dof_map) * self.dofs.take(indices=np.abs(dof_map)-1, axis=0)

        # evaluate basis functions (or derivatives)
        if d == 0:
            # values : nB x nE
            basis = self.fe_space.element.eval_basis(points, d=d)          # nB x nP
                
            #U = np.dot(values.T, basis)                                    # nE x nP
            U = (values[:,:,None] * basis[:,None,:]).sum(axis=0)           # nE x nP
        elif d == 1:
            # values : nB x nE
            dbasis_global = self.fe_space.eval_global_derivatives(points)  # nE x nB x nP x nD
                
            U = (values.T[:,:,None,None] * dbasis_global).sum(axis=1)      # nE x nP x nD
        else:
            raise NotImplementedError('Invalid derivation order ({})'.format(d))

        return U
