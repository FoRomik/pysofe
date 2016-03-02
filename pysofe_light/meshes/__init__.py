"""
Provides the data structures for the finite element meshes.

The main purpose of the mesh in the finite element environment 
is to discretize the spatial domain of the considered partial 
differential equation. Furthermore, it gives access to its 
topological entities such as vertices, edges, faces and cells
and provides methods for refinement.

The mesh data structure in Py\ **SOFE**\ *light* is represented
by the :py:class:`Mesh <pysofe_light.meshes.mesh.Mesh>` class and 
encapsulates the classes
:py:class:`MeshGeometry <pysofe_light.meshes.geometry.MeshGeometry>`
and :py:class:`MeshTopology <pysofe_light.meshes.topology.MeshTopology>`. 
It also has an instance of the 
:py:class:`ReferenceMap <pysofe_light.meshes.reference_map.ReferenceMap>` 
class as an attribute to connect the physical mesh entities with the
reference domain.
"""

import mesh
import geometry
import topology
import reference_map
import refinements

from mesh import Mesh
