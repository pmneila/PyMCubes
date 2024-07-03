
import numpy as np
from scipy import ndimage as ndi

__all__ = [
    'numpy_smooth',
]

FILTER = np.array([1, -2, 1], dtype=np.float64)

JACOBI_R_2D = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, -4, 0, 0],
    [1, -4, 0, -4, 1],
    [0, 0, -4, 0, 0],
    [0, 0, 1, 0, 0]
], dtype=np.float64)
JACOBI_D_2D = 1/12

JACOBI_R_3D = np.array([
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, -4, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
    [[0, 0, 1, 0, 0],
     [0, 0, -4, 0, 0],
     [1, -4, 0, -4, 1],
     [0, 0, -4, 0, 0],
     [0, 0, 1, 0, 0]],
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, -4, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]
], dtype=np.float64)

JACOBI_D_3D = 1/18


def signed_distance_function(binary_arr: np.ndarray) -> np.ndarray:

    arr = np.where(binary_arr > 0, 1.0, 0.0)
    dist_func = ndi.distance_transform_edt
    distance = np.where(
        binary_arr,
        dist_func(arr) - 0.5,
        -dist_func(1 - arr) + 0.5
    )
    return distance


def energy(arr: np.ndarray) -> np.ndarray:

    darr2 = [ndi.convolve1d(arr, FILTER, axis=i)**2 for i in range(arr.ndim)]
    return np.sum(darr2) / 2


def solve_jacobi(
        arr: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        max_iters: int = 500,
        jacobi_weight: float = 0.5
        ) -> np.ndarray:

    jacobi_d = JACOBI_D_2D if arr.ndim == 2 else JACOBI_D_3D
    jacobi_r = JACOBI_R_2D if arr.ndim == 2 else JACOBI_R_3D

    for it in range(max_iters):
        # energy_it = torch.sum(diff_energy(arr) * arr[2:-2, 2:-2, 2:-2]) / 2
        energy_it = energy(arr)
        print("Energy in iteration {}: {:.4g}".format(it, energy_it))

        r_arr = ndi.convolve(arr, jacobi_r, mode='nearest')
        arr_1 = - jacobi_d * r_arr
        arr = jacobi_weight * arr_1 + (1 - jacobi_weight) * arr

        arr = np.maximum(arr, lower_bound)
        arr = np.minimum(arr, upper_bound)

    return arr


def numpy_smooth(binary_array: np.ndarray, max_iters: int = 500) -> np.ndarray:

    arr = signed_distance_function(binary_array)

    upper_bound = np.where(arr < 0, arr, np.inf)
    lower_bound = np.where(arr > 0, arr, -np.inf)

    upper_bound[np.abs(upper_bound) < 1] = 0
    lower_bound[np.abs(lower_bound) < 1] = 0

    return solve_jacobi(arr, lower_bound, upper_bound, max_iters)
