"""
Provides some visualization capabilities.
"""

# IMPORTS
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
except ImportError as err:
    # Could not import pyplot
    # ... do some stuff here
    raise err

import numpy as np

import pysofe_light as pysofe

def show(obj, *args, **kwargs):
    """
    Wrapper function for the visualization of 
    various pysofe objects.

    Parameters
    ----------

    obj
        The pysofe object to visualize
    """

    # select appropriate visualizer and call its `show()` method
    if isinstance(obj, pysofe.elements.base.Element):
        V = ElementVisualizer()
        V.show(element=obj, **kwargs)
    elif isinstance(obj, pysofe.meshes.mesh.Mesh):
        V = MeshVisualizer()
        V.show(obj, *args, **kwargs)
#    elif isinstance(obj, (pysofe.spaces.functions.FEFunction,
#                          pysofe.spaces.functions.MeshFunction)):
#        V = FunctionVisualizer()
#        V.show(fnc=obj, **kwargs)
    else:
        raise NotImplementedError()

class Visualizer(object):
    """
    Base class for all visualizers.
    """

    def plot(self, *args, **kwargs):
        fig, axes = self._plot(*args, **kwargs)

        return fig, axes

    def _plot(self, *args, **kwargs):
        raise NotImplementedError()
    
    def show(self, *args, **kwargs):
        fig, axes = self.plot(*args, **kwargs)

        fig.show()

class MeshVisualizer(Visualizer):
    """
    Visualizes the :py:class:`pysofe.meshes.Mesh` class.
    """

    def _plot(self, mesh, *args, **kwargs):
        fontsize = kwargs.get('fontsize', 9)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)

        if mesh.dimension == 1:
            nodes = mesh.nodes[:,0]
            zeros = np.zeros_like(nodes)
            ax.plot(nodes, zeros, '-o')
        elif mesh.dimension == 2:
            cols = range(3)
            ax.triplot(mesh.nodes[:,0], mesh.nodes[:,1], np.asarray(mesh.cells[:,cols] - 1))
        else:
            raise NotImplementedError()
            
        # zoom out to make outer faces visible
        xlim = list(ax.get_xlim()); ylim = list(ax.get_ylim())
        xlim[0] -= 0.1; xlim[1] += 0.1
        ylim[0] -= 0.1; ylim[1] += 0.1
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    
        show_all = ('all' in args)
    
        # nodes
        if 'nodes' in args or show_all:
            for i in xrange(mesh.nodes.shape[0]):
                if mesh.dimension == 1:
                    ax.text(x=mesh.nodes[i,0], y=0., s=i+1,
                            color='red', fontsize=fontsize)
                elif mesh.dimension == 2:
                    ax.text(x=mesh.nodes[i,0], y=mesh.nodes[i,1], s=i+1,
                            color='red', fontsize=fontsize)
                else:
                    raise NotImplementedError()
                    
        # edges
        if 'edges' in args or show_all:
            edges = mesh.edges
            bary = 0.5 * mesh.nodes[edges - 1,:].sum(axis=1)
            for i in xrange(edges.shape[0]):
                if mesh.dimension == 1:
                    ax.text(x=bary[i,0], y=0, s=i+1,
                            color='green', fontsize=fontsize)
                elif mesh.dimension == 2:
                    ax.text(x=bary[i,0], y=bary[i,1], s=i+1,
                            color='green', fontsize=fontsize)

        # elements
        if mesh.dimension > 1 and ('cells' in args or show_all):
            cells = mesh.cells
            bary = mesh.nodes[cells - 1,:].sum(axis=1) / 3.
            for i in xrange(cells.shape[0]):
                ax.text(x=bary[i,0], y=bary[i,1], s=i+1,
                        color='blue', fontsize=fontsize)
        
        if 'local vertices' in args:
            cells = mesh.cells
            cell_nodes = mesh.nodes.take(cells - 1, axis=0)
            bary = cell_nodes.sum(axis=1) / 3.
            nE = cells.shape[0]
            
            # calculate positions where to put the local vertex numbers
            local_1 = cell_nodes[:,0] + 0.4 * (bary - cell_nodes[:,0])
            local_2 = cell_nodes[:,1] + 0.4 * (bary - cell_nodes[:,1])
            local_3 = cell_nodes[:,2] + 0.4 * (bary - cell_nodes[:,2])
            
            for i in xrange(nE):
                ax.text(x=local_1[i,0], y=local_1[i,1], s=1, color='red', fontsize=fontsize)
                ax.text(x=local_2[i,0], y=local_2[i,1], s=2, color='red', fontsize=fontsize)
                ax.text(x=local_3[i,0], y=local_3[i,1], s=3, color='red', fontsize=fontsize)

        return fig, ax

class ElementVisualizer(Visualizer):
    """
    Visualizes :py:class:`pysofe.elements.base.Element` classes.
    """

    def _plot(self, element, **kwargs):
        """
        Plots the basis function or their derivatives of the given element.
        
        Parameters
        ----------

        element : pysofe.base.Element
            The finite element of which to plot the basis functions

        codim : int
            The codimension of the entity for which to plot the respective basis functions

        d : int
            The derivation order

        indices : array_like
            Specify certain basis function to show

        resolution : int
            Resolution of the grid points for the plot

        typ : str
            The plotting type ('surface' or 'scatter')

        shadow : bool
            Whether to plot a shadow of the surface
        """
        
        # get arguments
        dim = kwargs.get('dim', element.dimension)
        d = kwargs.get('d', 0)
        indices = kwargs.get('indices', None)
        resolution = kwargs.get('resolution', 10*np.ceil(np.log(element.order+1)))
        typ = kwargs.get('typ', 'surface')
        shadow = kwargs.get('shadow', False)
        layout = kwargs.get('layout', None)

        if d != 0:
            raise NotImplementedError()

        if element.dimension > 2:
            raise NotImplementedError()

        codim = element.dimension - dim
        
        if element.dimension == 1:
            project = None
        elif element.dimension == 2:
            if codim == 0:
                project = '3d'
            elif codim == 1:
                project = None

        # create grid points at which to evaluate the basis functions
        ls = np.linspace(0., 1., num=resolution)

        if element.dimension == 1:
            points = ls
        
        elif element.dimension == 2:
            if codim == 0:
                X,Y = np.meshgrid(ls, ls)
                XY = np.vstack([np.hstack(X), np.hstack(Y)])
                points = XY.compress(XY.sum(axis=0) <= 1., axis=1)
            elif codim == 1:
                points = ls

        # evaluate all basis function at all points
        basis = element.eval_basis(points, d=d)    # nB x nP

        if indices is not None:
            assert hasattr(indices, '__iter__')
            indices = np.asarray(indices, dtype=int) - 1
            assert indices.min() >= 0

            basis = basis.take(indices, axis=0)
        else:
            indices = np.arange(np.size(basis, axis=0))

        # create a subplot for each basis function
        nB = np.size(basis, axis=0)

        fig = plt.figure()
        
        if layout is None:
            nB_2 = int(0.5*(nB+1))

            for i in xrange(1, nB_2+1):
                if codim == 0:
                    fig.add_subplot(nB_2,2,2*i-1, projection=project)
                    if 2*i <= nB:
                        fig.add_subplot(nB_2,2,2*i, projection=project)
                elif codim == 1:
                    fig.add_subplot(nB_2,2,2*i-1, projection=project)
                    if 2*i <= nB:
                        fig.add_subplot(nB_2,2,2*i, projection=project)
            
            
        else:
            assert 1 <= len(layout) <= 2
        
            if len(layout) == 1:
                layout = (1,layout[0])
        
            assert np.multiply(*layout) >= nB
        
            for j in xrange(nB):
                if codim == 0:
                    fig.add_subplot(layout[0], layout[1], j+1, projection=project)
                elif codim == 1:
                    fig.add_subplot(layout[0], layout[1], j+1, projection=project)


        if element.dimension == 1:
            for i in xrange(nB):
                fig.axes[i].plot(points.ravel(), basis[i].ravel())
                #fig.axes[i].set_title(r"$\varphi_{{ {} }}$".format(i+1), fontsize=32)
                fig.axes[i].set_title(r"$\varphi_{{ {} }}$".format(indices[i]+1), fontsize=32)
        
        elif element.dimension == 2:
            if codim == 0:
                for i in xrange(nB):
                    if typ == 'scatter':
                        fig.axes[i].scatter(points[0], points[1], basis[i])
                    elif typ == 'surface':
                        fig.axes[i].plot_trisurf(points[0], points[1], basis[i],
                                                 cmap=cm.jet, linewidth=0., antialiased=False)
                        if shadow:
                            c = fig.axes[i].tricontourf(points[0], points[1], basis[i],
                                                        zdir='z', offset=0., colors='gray')
                    fig.axes[i].autoscale_view(True,True,True)
                    #fig.axes[i].set_title(r"$\varphi_{{ {} }}$".format(i+1), fontsize=32)
                    fig.axes[i].set_title(r"$\varphi_{{ {} }}$".format(indices[i]+1), fontsize=32)
            elif codim == 1:
                for i in xrange(nB):
                    fig.axes[i].plot(points.ravel(), basis[i].ravel())
                    #fig.axes[i].set_title(r"$\psi_{{ {} }}$".format(i+1), fontsize=32)
                    fig.axes[i].set_title(r"$\psi_{{ {} }}$".format(indices[i]+1), fontsize=32)
    
        return fig, fig.axes
