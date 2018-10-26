# PyMCubes

`PyMCubes` is an implementation of the marching cubes algorithm to extract
isosurfaces from volumetric data. The volumetric data can be given as a
three-dimensional `NumPy` array or as a Python function ``f(x, y, z)``. The first
option is much faster, but it requires more memory and becomes unfeasible for
very large volumes.

`PyMCubes` also provides a function to export the results of the marching cubes as
COLLADA ``(.dae)`` files. This requires the [`PyCollada`](https://github.com/pycollada/pycollada) library.

## Installation

Use `pip`:

```
$ pip install --upgrade PyMCubes
```

## Example

The following example creates a `NumPy` volume with spherical isosurfaces and
extracts one of them (i.e., a sphere) with `PyMCubes`. The result is exported to
`sphere.dae`:

```Python
  >>> import numpy as np
  >>> import mcubes
  
  # Create a data volume (30 x 30 x 30)
  >>> X, Y, Z = np.mgrid[:30, :30, :30]
  >>> u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2
  
  # Extract the 0-isosurface
  >>> vertices, triangles = mcubes.marching_cubes(u, 0)
  
  # Export the result to sphere.dae
  >>> mcubes.export_mesh(vertices, triangles, "sphere.dae", "MySphere")
```

The second example uses a function to represent the volume instead of a NumPy array.

```Python
  >>> import numpy as np
  >>> import mcubes
  
  # Create the volume
  >>> f = lambda x, y, z: x**2 + y**2 + z**2
  
  # Extract the 16-isosurface
  >>> vertices, triangles = mcubes.marching_cubes_func((-10,-10,-10), (10,10,10),
  ... 100, 100, 100, f, 16)
  
  # Export the result to sphere2.dae
  >>> mcubes.export_mesh(vertices, triangles, "sphere2.dae", "MySphere")
```

Note that using a function to represent the volumetric data is **much** slower than using a `NumPy` array.
