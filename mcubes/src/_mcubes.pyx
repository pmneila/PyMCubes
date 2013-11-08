
# distutils: language = c++
# cython: embedsignature = True

import numpy as np

# Define PY_ARRAY_UNIQUE_SYMBOL
cdef extern from "pyarray_symbol.h":
    pass

cimport numpy as np

np.import_array()

cdef extern from "pywrapper.h":
    cdef object c_marching_cubes "marching_cubes"(np.ndarray, double) except +

def marching_cubes(np.ndarray volume, float isovalue):
    
    verts, faces = c_marching_cubes(volume, isovalue)
    verts.shape = (-1, 3)
    faces.shape = (-1, 3)
    return verts, faces
