from IPython import embed as IPS
import matplotlib.pyplot as plt

from pysofe.meshes.generation.distance_functions import *
from pysofe.meshes.generation.generators import MeshGenerator, uniform_edge_lengths

sdf0 = DSphere(centre=[0, 0], radius=1)
sdf1 = DSphere(centre=[1, 0], radius=1)
sdf2 = DOrthotope(limits=[(0.5,1.5), (0.5,1.5)])
sdf3 = DCircle()
sdf4 = DRectangle()

#sdf = DUnion(sdf0, sdf1)
#sdf = DIntersection(sdf0, sdf1)
#sdf = DDifference(sdf0, sdf1)
#sdf = DDifference(sdf2, sdf0)
#sdf = DUnion(sdf4, sdf2)

#elf = uniform_edge_lengths

sdf = DCircle()
elf = lambda x: 0.1 - sdf(x)

mg = MeshGenerator(sdf, elf)

IPS()
