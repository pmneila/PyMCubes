
========================================================
PyMCubes - Marching cubes (and related tools) for Python
========================================================

PyMCubes is an implementation of the marching cubes algorithm to extract
isosurfaces from volumetric data. The current implementation only accepts
three-dimensional NumPy arrays as representation of volumetric data.

PyMCubes also provides a function to export the results of the marching cubes as
COLLADA ``(.dae)`` files. This requires the
`PyCollada <https://github.com/pycollada/pycollada>`_ library.

Example
=======

The following example creates a data volume with spherical isosurfaces and
extracts one of them (i.e., a sphere) with PyMCubes. The result is exported as
``sphere.dae``::

  >>> import numpy as np
  >>> import mcubes
  
  # Create a data volume (30 x 30 x 30)
  >>> X, Y, Z = np.mgrid[:30, :30, :30]
  >>> u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2
  
  # Extract the 0-isosurface
  >>> vertices, triangles = mcubes.marching_cubes(u, 0)
  
  # Export the result to sphere.dae
  >>> mcubes.export_mesh(vertices, triangles, "sphere.dae", "mysphere")
