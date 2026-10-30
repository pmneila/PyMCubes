
# distutils: language = c++
# cython: embedsignature = True

# from libcpp.vector cimport vector
import numpy as np

# Define PY_ARRAY_UNIQUE_SYMBOL
cdef extern from "pyarray_symbol.h":
    pass

cimport numpy as np

np.import_array()

cdef extern from "pywrapper.h":
    cdef object c_marching_cubes "marching_cubes"(np.ndarray, double, int) except +
    cdef object c_marching_cubes_func "marching_cubes_func"(tuple, tuple, int, int, int, object, double, int) except +

def marching_cubes(np.ndarray volume, float isovalue, int num_threads=0):
    """Extract the isosurface of a 3D volume.

    ``num_threads`` selects how many threads the algorithm may use. The default
    of 0 uses one thread per CPU core, and 1 keeps it single-threaded. The
    result does not depend on this value.
    """

    if num_threads < 0:
        raise ValueError("num_threads cannot be negative")

    verts, faces = c_marching_cubes(volume, isovalue, num_threads)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces

def marching_cubes_func(tuple lower, tuple upper, int numx, int numy, int numz, object f, double isovalue, int num_threads=0):
    """Extract the isosurface of a volume given as a function ``f(x, y, z)``.

    ``num_threads`` selects how many threads the algorithm may use. The default
    of 0 uses one thread per CPU core, and 1 keeps it single-threaded. The
    result does not depend on this value.
    """
    
    if any(l_i >= u_i for l_i, u_i in zip(lower, upper)):
        raise ValueError("lower coordinates cannot be larger than upper coordinates")
    
    if numx < 2 or numy < 2 or numz < 2:
        raise ValueError("numx, numy, numz cannot be smaller than 2")

    if num_threads < 0:
        raise ValueError("num_threads cannot be negative")

    verts, faces = c_marching_cubes_func(lower, upper, numx, numy, numz, f, isovalue, num_threads)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces
