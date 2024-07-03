# -*- coding: utf-8 -*-

"""
Utilities for smoothing the 0.5 level-set of binary arrays.
"""

import logging
from typing import Tuple

import numpy as np
from scipy import sparse
from scipy import ndimage as ndi

__all__ = [
    'smooth',
    'smooth_constrained',
    'smooth_gaussian',
    'signed_distance_function'
]


def _build_variable_indices(band: np.ndarray) -> np.ndarray:
    num_variables = np.count_nonzero(band)
    variable_indices = np.full(band.shape, -1, dtype=np.int32)
    variable_indices[band] = np.arange(num_variables)
    return variable_indices


def _buildq3d(variable_indices: np.ndarray):
    """
    Builds the filterq matrix for the given variables.
    """

    num_variables = variable_indices.max() + 1
    filterq = sparse.lil_matrix((3*num_variables, num_variables))

    # Pad variable_indices to simplify out-of-bounds accesses
    variable_indices = np.pad(
        variable_indices,
        [(0, 1), (0, 1), (0, 1)],
        mode='constant',
        constant_values=-1
    )

    coords = np.nonzero(variable_indices >= 0)
    for count, (i, j, k) in enumerate(zip(*coords)):

        assert variable_indices[i, j, k] == count

        filterq[3*count, count] = -2
        neighbor = variable_indices[i-1, j, k]
        if neighbor >= 0:
            filterq[3*count, neighbor] = 1
        else:
            filterq[3*count, count] += 1

        neighbor = variable_indices[i+1, j, k]
        if neighbor >= 0:
            filterq[3*count, neighbor] = 1
        else:
            filterq[3*count, count] += 1

        filterq[3*count+1, count] = -2
        neighbor = variable_indices[i, j-1, k]
        if neighbor >= 0:
            filterq[3*count+1, neighbor] = 1
        else:
            filterq[3*count+1, count] += 1

        neighbor = variable_indices[i, j+1, k]
        if neighbor >= 0:
            filterq[3*count+1, neighbor] = 1
        else:
            filterq[3*count+1, count] += 1

        filterq[3*count+2, count] = -2
        neighbor = variable_indices[i, j, k-1]
        if neighbor >= 0:
            filterq[3*count+2, neighbor] = 1
        else:
            filterq[3*count+2, count] += 1

        neighbor = variable_indices[i, j, k+1]
        if neighbor >= 0:
            filterq[3*count+2, neighbor] = 1
        else:
            filterq[3*count+2, count] += 1

    filterq = filterq.tocsr()
    return filterq.T.dot(filterq)


def _buildq2d(variable_indices: np.ndarray):
    """
    Builds the filterq matrix for the given variables.

    Version for 2 dimensions.
    """

    num_variables = variable_indices.max() + 1
    filterq = sparse.lil_matrix((3*num_variables, num_variables))

    # Pad variable_indices to simplify out-of-bounds accesses
    variable_indices = np.pad(
        variable_indices,
        [(0, 1), (0, 1)],
        mode='constant',
        constant_values=-1
    )

    coords = np.nonzero(variable_indices >= 0)
    for count, (i, j) in enumerate(zip(*coords)):

        assert variable_indices[i, j] == count

        filterq[2*count, count] = -2
        neighbor = variable_indices[i-1, j]
        if neighbor >= 0:
            filterq[2*count, neighbor] = 1
        else:
            filterq[2*count, count] += 1

        neighbor = variable_indices[i+1, j]
        if neighbor >= 0:
            filterq[2*count, neighbor] = 1
        else:
            filterq[2*count, count] += 1

        filterq[2*count+1, count] = -2
        neighbor = variable_indices[i, j-1]
        if neighbor >= 0:
            filterq[2*count+1, neighbor] = 1
        else:
            filterq[2*count+1, count] += 1

        neighbor = variable_indices[i, j+1]
        if neighbor >= 0:
            filterq[2*count+1, neighbor] = 1
        else:
            filterq[2*count+1, count] += 1

    filterq = filterq.tocsr()
    return filterq.T.dot(filterq)


def _jacobi(
        filterq,
        x0: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        max_iters: int = 10,
        rel_tol: float = 1e-6,
        weight: float = 0.5):
    """Jacobi method with constraints."""

    jacobi_r = sparse.lil_matrix(filterq)
    shp = jacobi_r.shape
    jacobi_d = 1.0 / filterq.diagonal()
    jacobi_r.setdiag((0,) * shp[0])
    jacobi_r = jacobi_r.tocsr()

    x = x0

    # We check the stopping criterion each 10 iterations
    check_each = 10
    cum_rel_tol = 1 - (1 - rel_tol)**check_each

    energy_now = np.dot(x, filterq.dot(x)) / 2
    logging.debug("Energy at iter %d: %.6g", 0, energy_now)
    for i in range(max_iters):

        x_1 = - jacobi_d * jacobi_r.dot(x)
        x = weight * x_1 + (1 - weight) * x

        # Constraints.
        x = np.maximum(x, lower_bound)
        x = np.minimum(x, upper_bound)

        # Stopping criterion
        if (i + 1) % check_each == 0:
            # Update energy
            energy_before = energy_now
            energy_now = np.dot(x, filterq.dot(x)) / 2

            logging.debug("Energy at iter %d: %.6g", i + 1, energy_now)

            # Check stopping criterion
            cum_rel_improvement = (energy_before - energy_now) / energy_before
            if cum_rel_improvement < cum_rel_tol:
                break

    return x


def signed_distance_function(
        levelset: np.ndarray,
        band_radius: int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the distance to the 0.5 levelset of a function, the mask of the
    border (i.e., the nearest cells to the 0.5 level-set) and the mask of the
    band (i.e., the cells of the function whose distance to the 0.5 level-set
    is less of equal to `band_radius`).
    """

    binary_array = np.where(levelset > 0, True, False)

    # Compute the band and the border.
    dist_func = ndi.distance_transform_edt
    distance = np.where(
        binary_array,
        dist_func(binary_array) - 0.5,
        -dist_func(~binary_array) + 0.5
    )
    border = np.abs(distance) < 1
    band = np.abs(distance) <= band_radius

    return distance, border, band


def smooth_constrained(
        binary_array: np.ndarray,
        band_radius: int = 4,
        max_iters: int = 500,
        rel_tol: float = 1e-6
        ) -> np.ndarray:
    """
    Implementation of the smoothing method from

    "Surface Extraction from Binary Volumes with Higher-Order Smoothness"
    Victor Lempitsky, CVPR10
    """

    # # Compute the distance map, the border and the band.
    logging.info("Computing distance transform...")
    distance, _, band = signed_distance_function(binary_array, band_radius)

    variable_indices = _build_variable_indices(band)

    # Compute filterq.
    logging.info("Building matrix filterq...")
    if binary_array.ndim == 3:
        filterq = _buildq3d(variable_indices)
    elif binary_array.ndim == 2:
        filterq = _buildq2d(variable_indices)
    else:
        raise ValueError("binary_array.ndim not in [2, 3]")

    # Initialize the variables.
    res = np.asarray(distance, dtype=np.double)
    x = res[band]
    upper_bound = np.where(x < 0, x, np.inf)
    lower_bound = np.where(x > 0, x, -np.inf)

    upper_bound[np.abs(upper_bound) < 1] = 0
    lower_bound[np.abs(lower_bound) < 1] = 0

    # Solve.
    logging.info("Minimizing energy...")
    x = _jacobi(
        filterq=filterq,
        x0=x,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        max_iters=max_iters,
        rel_tol=rel_tol
    )

    res[band] = x
    return res


def smooth_gaussian(binary_array: np.ndarray, sigma: float = 3) -> np.ndarray:
    vol = np.float64(binary_array) - 0.5
    return ndi.gaussian_filter(vol, sigma=sigma)


def smooth(
        binary_array: np.ndarray,
        method: str = 'auto',
        **kwargs
        ) -> np.ndarray:
    """
    Smooths the 0.5 level-set of a binary array. Returns a floating-point
    array with a smoothed version of the original level-set in the 0 isovalue.

    This function can apply two different methods:

    - A constrained smoothing method which preserves details and fine
      structures, but it is slow and requires a large amount of memory. This
      method is recommended when the input array is small (smaller than
      (500, 500, 500)).
    - A Gaussian filter applied over the binary array. This method is fast, but
      not very precise, as it can destroy fine details. It is only recommended
      when the input array is large and the 0.5 level-set does not contain
      thin structures.

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
        Size of the Gaussian filter (default 3).

    Parameters for 'constrained'
    ----------------------------
    max_iters : positive integer
        Number of iterations of the constrained optimization method
        (default 500).
    rel_tol: float
        Relative tolerance as a stopping criterion (default 1e-6).

    Output
    ------
    res : ndarray
        Floating-point array with a smoothed 0 level-set.
    """

    binary_array = np.asarray(binary_array)

    if method == 'auto':
        if binary_array.size > 500**3:
            method = 'gaussian'
        else:
            method = 'constrained'

    if method == 'gaussian':
        return smooth_gaussian(binary_array, **kwargs)

    if method == 'constrained':
        return smooth_constrained(binary_array, **kwargs)

    raise ValueError("Unknown method '{}'".format(method))
