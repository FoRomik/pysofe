from IPython import embed as IPS
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial import Delaunay

import pysofe
from pysofe.meshes.generation.distance_functions import *
from pysofe.meshes.generation.generators import MeshGenerator, uniform_edge_lengths

sdf_sphere = DSphere(centre=[0.5, 0.5], radius=0.25)
sdf_rect = DRectangle0(xlim=[0,1], ylim=[0,1])

uelf = uniform_edge_lengths
delf = lambda p: 1 + 8 * np.abs(sdf_sphere(p))

mg = MeshGenerator(sdf=sdf_rect,
                   elf=uelf)

mgi = MeshGenerator(sdf=sdf_rect,
                    elf=delf,
                    sdf_int=sdf_sphere)

h0 = 0.02

n = np.arange(0,1+3*h0,3*h0)
z = np.zeros_like(n)
o = np.ones_like(n)

f = np.hstack([[z,n],[o,n],[n,z],[n,o]]).T
f = pysofe.utils.unique_rows(f)

#q, w = mgi.generate(h0=h0, fixed_nodes=f)
nodes, cells = mgi.generate(0.02)

IPS()

# # make rectangular mesh periodic
# # ------------------------------

# # map boundary points to true boundary
# left = nodes[:,0] < sdf_rect.limits[0,0] + mgi._gtol
# right = nodes[:,0] > sdf_rect.limits[0,1] - mgi._gtol
# bottom = nodes[:,1] < sdf_rect.limits[1,0] + mgi._gtol
# top = nodes[:,1] > sdf_rect.limits[1,1] - mgi._gtol

# inner = np.logical_not(left + right + bottom + top)

# nodes[left,0] = sdf_rect.limits[0,0]
# nodes[right,0] = sdf_rect.limits[0,1]
# nodes[bottom,1] = sdf_rect.limits[1,0]
# nodes[top,1] = sdf_rect.limits[1,1]

# lr1 = np.unique(np.hstack([nodes[left,1], nodes[right,1]]))
# bt0 = np.unique(np.hstack([nodes[bottom,0], nodes[top,0]]))

# # handle nodes too close now
# lr1_imask = (np.diff(lr1) > h0).nonzero()[0]
# bt0_imask = (np.diff(bt0) > h0).nonzero()[0]

# lr1 = lr1.take(np.hstack([lr1_imask, lr1.size-1]))
# bt0 = bt0.take(np.hstack([bt0_imask, bt0.size-1]))

# new_left_nodes = np.vstack([sdf_rect.limits[0,0] * np.ones_like(lr1), lr1]).T
# new_right_nodes = np.vstack([sdf_rect.limits[0,1] * np.ones_like(lr1), lr1]).T
# new_bottom_nodes = np.vstack([bt0, sdf_rect.limits[1,0] * np.ones_like(bt0)]).T
# new_top_nodes = np.vstack([bt0, sdf_rect.limits[1,1] * np.ones_like(bt0)]).T

# boundary_nodes = np.vstack([new_left_nodes,
#                             new_right_nodes,
#                             new_bottom_nodes,
#                             new_top_nodes])

# inner_nodes = nodes.compress(inner, axis=0)

# new_nodes = np.vstack([pysofe.utils.unique_rows(boundary_nodes),
#                        inner_nodes])

# new_cells = Delaunay(new_nodes).simplices.astype('int') + 1

