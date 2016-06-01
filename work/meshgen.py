from IPython import embed as IPS
import matplotlib.pyplot as plt

import numpy as np

from pysofe.meshes.generation.distance_functions import *
from pysofe.meshes.generation.generators import MeshGenerator, uniform_edge_lengths

sdf_sphere = DSphere(centre=[0.5, 0.5], radius=0.25)
sdf_rect = DRectangle0(xlim=[0,1], ylim=[0,1])

uelf = uniform_edge_lengths

mg = MeshGenerator(sdf=sdf_rect,
                   elf=uelf)

mgi = MeshGenerator(sdf=sdf_rect,
                    elf=uelf,
                    sdf_int=sdf_sphere)

IPS()
