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

class Visualizer(object):
    '''
    Base class for all visualizers.
    '''

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

    def _plot(self, mesh, *args):
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
        cols = range(3)
        ax.triplot(mesh.nodes[:,0], mesh.nodes[:,1], np.asarray(mesh.cells[:,cols] - 1))
    
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
                ax.text(x=mesh.nodes[i,0], y=mesh.nodes[i,1], s=i+1, color='red')
            
        # edges
        if 'edges' in args or show_all:
            edges = mesh.edges
            bary = 0.5 * mesh.nodes[edges - 1,:].sum(axis=1)
            for i in xrange(edges.shape[0]):
                ax.text(x=bary[i,0], y=bary[i,1], s=i+1, color='green')

        # elements
        if 'cells' in args or show_all:
            cells = mesh.cells
            bary = mesh.nodes[cells - 1,:].sum(axis=1) / 3.
            for i in xrange(cells.shape[0]):
                ax.text(x=bary[i,0], y=bary[i,1], s=i+1, color='blue')
        
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
                ax.text(x=local_1[i,0], y=local_1[i,1], s=1, color='red')
                ax.text(x=local_2[i,0], y=local_2[i,1], s=2, color='red')
                ax.text(x=local_3[i,0], y=local_3[i,1], s=3, color='red')

        return fig, ax
