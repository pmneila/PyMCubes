# -*- coding: utf-8 -*-

"""
Implementation of the smoothing method from

"Surface Extraction from Binary Volumes with Higher-Order Smoothness"
Victor Lempitsky, CVPR10

2D example
----------

Create a binary embedding function. For example:

>>> import numpy as np
>>> X,Y = np.mgrid[:50, :50]
>>> dist = np.sqrt((X-25)**2 + (Y-25)**2)
>>> u = (dist - 15) < 0

u is a binary function with 1 inside a circle and 0 outside it.
Smooth the 0.5 levelset surface with:

>>> import mcubes
>>> usmooth = mcubes.smooth(u, 200, verbose=True)

You can compare the results plotting the 0.5-levelset of u
and the 0-levelset of usmooth:

>>> contour(u, [0.5], colors='b')
>>> contour(u, [0], colors='r')
>>> axis('image')

3D example
----------

First, create a binary embedding function. For example,
a sphere:

>>> X,Y,Z = np.mgrid[:100,:100,:100]
>>> dist = np.sqrt((X-50)**2 + (Y-50)**2 + (Z-50)**2)
>>> u = dist-30 < 0

u is a binary function with 1 inside the sphere and 0 outside.
Smooth the 0.5 levelset surface with

>>> import mcubes
>>> usmooth = mcubes.smooth(u, 300, verbose=True)

You can compare the original volume and the result by extracting
the 0.5-levelset of u and the 0-levelset of usmooth with marching cubes:

>>> vertices, triangles = mcubes.marching_cubes(u, 0.5)
>>> mcubes.export_mesh(vertices, triangles, 'sphere1.dae', 'Sphere_nosmooth')
>>> vertices, triangles = mcubes.marching_cubes(usmooth, 0)
>>> mcubes.export_mesh(vertices, triangles, 'sphere2.dae', 'Sphere_smooth')

"""

__author__ = "Pablo Márquez Neila"

import logging

import numpy as np
from scipy import sparse
from scipy import ndimage as ndi

__all__ = ['smooth']


def build_variable_indices(band):
    num_variables = np.count_nonzero(band)
    variable_indices = np.full(band.shape, -1, dtype=np.int_)
    variable_indices[band] = np.arange(num_variables)
    return variable_indices


def buildQ3d(variable_indices):
    """
    Builds the Q matrix for the variables
    in the band given as the boolean matrix `band`.
    """
    
    num_variables = variable_indices.max() + 1
    Q = sparse.lil_matrix((3*num_variables, num_variables))
    
    # Pad variable_indices to simplify out-of-bounds accesses
    variable_indices = np.pad(
        variable_indices,
        [(0, 1), (0, 1), (0, 1)],
        mode='constant',
        constant_values=-1
    )
    
    coords = np.nonzero(variable_indices >= 0)
    for count, (i, j, k) in enumerate(zip(*coords)):
        
        assert(variable_indices[i, j, k] == count)
        
        Q[3*count, count] = -2
        neighbor = variable_indices[i-1, j, k]
        if neighbor >= 0:
            Q[3*count, neighbor] = 1
        else:
            Q[3*count, count] += 1
        
        neighbor = variable_indices[i+1, j, k]
        if neighbor >= 0:
            Q[3*count, neighbor] = 1
        else:
            Q[3*count, count] += 1
        
        Q[3*count+1, count] = -2
        neighbor = variable_indices[i , j-1, k]
        if neighbor >= 0:
            Q[3*count+1, neighbor] = 1
        else:
            Q[3*count+1, count] += 1
        
        neighbor = variable_indices[i, j+1, k]
        if neighbor >= 0:
            Q[3*count+1, neighbor] = 1
        else:
            Q[3*count+1, count] += 1
        
        Q[3*count+2, count] = -2
        neighbor = variable_indices[i, j, k-1]
        if neighbor >= 0:
            Q[3*count+2, neighbor] = 1
        else:
            Q[3*count+2, count] += 1
        
        neighbor = variable_indices[i, j, k+1]
        if neighbor >= 0:
            Q[3*count+2, neighbor] = 1
        else:
            Q[3*count+2, count] += 1
    
    Q = Q.tocsr()
    return Q.T.dot(Q)


def buildQ2d(variable_indices):
    """
    Builds the Q matrix for the variables
    in the band given as the boolean matrix `band`.
    
    Version for 2 dimensions.
    """
    
    num_variables = variable_indices.max() + 1
    Q = sparse.lil_matrix((3*num_variables, num_variables))
    
    # Pad variable_indices to simplify out-of-bounds accesses
    variable_indices = np.pad(
        variable_indices,
        [(0, 1), (0, 1)],
        mode='constant',
        constant_values=-1
    )
    
    coords = np.nonzero(variable_indices >= 0)
    for count, (i, j) in enumerate(zip(*coords)):
        assert(variable_indices[i, j] == count)
        
        Q[2*count, count] = -2
        neighbor = variable_indices[i-1, j]
        if neighbor >= 0:
            Q[2*count, neighbor] = 1
        else:
            Q[2*count, count] += 1
        
        neighbor = variable_indices[i+1, j]
        if neighbor >= 0:
            Q[2*count, neighbor] = 1
        else:
            Q[2*count, count] += 1
        
        Q[2*count+1, count] = -2
        neighbor = variable_indices[i, j-1]
        if neighbor >= 0:
            Q[2*count+1, neighbor] = 1
        else:
            Q[2*count+1, count] += 1
        
        neighbor = variable_indices[i, j+1]
        if neighbor >= 0:
            Q[2*count+1, neighbor] = 1
        else:
            Q[2*count+1, count] += 1
    
    Q = Q.tocsr()
    return Q.T.dot(Q)


def get_distance(f, band_radius):
    """
    Return the distance to the 0.5 levelset of a function.
    
    Besides, get_distance returns the mask of the border,
    i.e., the nearest cells to the 0.5 level-set, and the
    mask of the band, ie., the cells of the function whose
    distance to the 0.5 level-set is less of equal to band_radius.
    """
    
    # Prepare the embedding function.
    f = np.double(np.copy(f))
    f[f <= 0] = 0
    f[f > 0] = 1
    
    # Compute the band and the border.
    dist_func = ndi.distance_transform_edt
    distance = np.maximum(dist_func(f) - 0.5, dist_func(1-f) - 0.5)
    border = (distance < 1)
    band = (distance <= band_radius)
    distance[f == 0] *= -1
    
    return distance, border, band


def jacobi(Q, x, l, u, num_iters = 10, w=0.5):
    """Jacobi method."""
    
    R = sparse.lil_matrix(Q)
    shp = R.shape
    D = 1.0 / Q.diagonal()
    R.setdiag((0,) * shp[0])
    R = R.tocsr()
    
    for i in range(num_iters):
        x_1 = - D * R.dot(x)
        x_1 = w * x_1 + (1-w) * x
        
        # Constraints.
        x_1 = np.where(x_1 > l, x_1, l)
        x = np.where(x_1 < u, x_1, u)
        
        energy = np.dot(x, Q.dot(x))
        logging.debug("Energy in iteration {}: {:.4g}".format(i, energy))
    
    return x


def smooth_constrained(binary_array, num_iters=250, band_radius=4):
    """
    Smooth the zero level-set of a binary array.
    
    The smoothing is performed in a narrow band near the zero
    level-set.
    """
    
    # # Compute the distance map, the border and the band.
    logging.info("Computing distance transform...")
    distance, border, band = get_distance(binary_array, band_radius)
    
    variable_indices = build_variable_indices(band)
    
    # Compute Q.
    logging.info("Building matrix Q...")
    if binary_array.ndim == 3:
        Q = buildQ3d(variable_indices)
    elif binary_array.ndim == 2:
        Q = buildQ2d(variable_indices)
    else:
        raise ValueError("binary_array.ndim not in [2, 3]")
    
    # Initialize the variables.
    res = np.asarray(distance, dtype=np.double)
    x = res[band]
    border_variables = variable_indices[border]
    u = np.where(x < 0, x, np.inf)
    u[border_variables] = np.where(x[border_variables] < 0, 0.0, np.inf)
    l = np.where(x > 0, x, -np.inf)
    l[border_variables] = np.where(x[border_variables] > 0, 0.0, -np.inf)
    
    # Solve.
    logging.info("Minimizing energy...")
    x = jacobi(Q, x, l, u, num_iters)
    
    res[band] = x
    return res


def smooth(binary_array, method='auto', **kwargs):
    """
    Smooths the 0.5 level-set of a binary array. Returns a floating-point
    array with a smoothed version of the original level-set in the 0 isovalue.
    
    The smoothing can be performed with two different methods:
    
    - A Gaussian filter applied over the binary array. This method is fast, but
      not very precise, as it can destroy fine details. It is only recommended
      when the input array is large and the 0.5 level-set does not contain
      thin structures. 
    - A constrained smoothing method which preserves details and fine
      structures, but it is slow and requires a large amount of memory. This
      method is recommended when the input array is small (smaller than
      (150, 150, 150)).
    
    Parameters
    ----------
    binary_array : ndarray
        Input binary array with the 0.5 level-set to smooth.
    method : str, one of ['auto', 'gaussian', 'constrained']
        Smoothing method. If 'auto' is given, the method will be automatically
        chosen based on the size of `binary_array`.
    
    Parameters for 'gaussian'
    -------------------------
    sigma : float
        Size of the Gaussian filter. 3 by default.
    
    Parameters for 'constrained'
    ----------------------------
    num_iters : positive integer
        Number of iterations of the constrained optimization method.
        250 by default.
    
    Output
    ------
    res : ndarray
        Floating-point array with a smoothed 0 level-set.
    """
    
    binary_array = np.array(binary_array)
    
    if method == 'auto':
        if binary_array.size > 150**3:
            method = 'gaussian'
        else:
            method = 'constrained'
    
    if method == 'gaussian':
        params = {'sigma': 3}
        params.update(kwargs)
        
        vol = np.float_(binary_array) - 0.5
        return ndi.gaussian_filter(vol, **params)
    
    if method == 'constrained':
        params = {'num_iters': 250}
        params.update(kwargs)
        return smooth_constrained(binary_array, **params)
