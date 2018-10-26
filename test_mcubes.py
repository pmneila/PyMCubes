
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import mcubes


def test_empty():
    
    levelset = np.zeros((50, 50, 50))
    vertices, triangles = mcubes.marching_cubes(levelset, 0.5)
    
    assert len(vertices) == len(triangles) == 0


def test_sphere():
    x, y, z = np.mgrid[:100, :100, :100]
    u = (z - 50)**2 + (y - 50)**2 + (z - 50)**2 - 25**2
    
    def func(x, y, z):
        return u[int(x), int(y), int(z)]
    
    vertices1, triangles1 = mcubes.marching_cubes(u, 0.0)
    vertices2, triangles2 = mcubes.marching_cubes_func(
        (0, 0, 0),
        (99, 99, 99),
        100, 100, 100,
        func, 0.0
    )
    
    assert_allclose(vertices1, vertices2)
    assert_array_equal(triangles1, triangles2)


def test_export():
    
    u = np.zeros((10, 10, 10))
    u[2:-2, 2:-2, 2:-2] = 1.0
    vertices, triangles = mcubes.marching_cubes(u, 0.5)
    
    mcubes.export_obj(vertices, triangles, "output/test.obj")
    mcubes.export_off(vertices, triangles, "output/test.off")
    mcubes.export_mesh(vertices, triangles, "output/test.dae")
