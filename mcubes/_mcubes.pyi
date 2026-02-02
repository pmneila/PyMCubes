from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray


def marching_cubes(
    volume: NDArray[np.generic],
    isovalue: float,
) -> Tuple[NDArray[np.float64], NDArray[np.uintp]]: ...


def marching_cubes_func(
    lower: Tuple[float, float, float],
    upper: Tuple[float, float, float],
    numx: int,
    numy: int,
    numz: int,
    f: Callable[[float, float, float], float],
    isovalue: float,
) -> Tuple[NDArray[np.float64], NDArray[np.uintp]]: ...
