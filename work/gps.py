from IPython import embed as IPS

import pysofe

mesh = pysofe.UnitSquareMesh(32, 32)
bary = mesh.nodes.take(mesh.cells-1, axis=0).mean(axis=1)

IPS()
