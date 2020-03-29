

import pytest

import numpy as np

import mcubes


def test_sphere():

    # Create sphere with radius 25 centered at (50, 50, 50)
    x, y, z = np.mgrid[:100, :100, :100]
    levelset = np.sqrt((x - 50)**2 + (y - 50)**2 + (z - 50)**2) - 25

    # vertices, triangles = mcubes.marching_cubes(levelset, 0)
    # mcubes.export_obj(vertices, triangles, 'sphere1.obj')

    binary_levelset = levelset > 0
    smoothed_levelset = mcubes.smooth(
        binary_levelset,
        method='constrained',
        max_iters=500,
        rel_tol=1e-4
    )

    vertices, _ = mcubes.marching_cubes(smoothed_levelset, 0.0)

    # Check all vertices have same distance to (50, 50, 50)
    dist = np.sqrt(np.sum((vertices - [50, 50, 50])**2, axis=1))

    assert dist.min() > 24.5 and dist.max() < 25.5
    assert np.all(np.abs(smoothed_levelset - levelset) < 1)
    assert np.all((smoothed_levelset > 0) == binary_levelset)


def test_gaussian_smoothing():

    # Create sphere with radius 25 centered at (50, 50, 50)
    x, y, z = np.mgrid[:100, :100, :100]
    levelset = np.sqrt((x - 50)**2 + (y - 50)**2 + (z - 50)**2) - 25

    binary_levelset = levelset > 0
    smoothed_levelset = mcubes.smooth(
        binary_levelset,
        method='gaussian',
        sigma=3
    )

    vertices, _ = mcubes.marching_cubes(smoothed_levelset, 0.0)

    # Check all vertices have same distance to (50, 50, 50)
    dist = np.sqrt(np.sum((vertices - [50, 50, 50])**2, axis=1))
    assert dist.min() > 24 and dist.max() < 25


def test_wrong_ndim():

    binary_levelset = np.random.uniform(size=(10)) < 0.5

    with pytest.raises(ValueError):
        mcubes.smooth(
            binary_levelset,
            method='constrained',
            max_iters=500,
            rel_tol=1e-4
        )

    binary_levelset = np.random.uniform(size=(10, 10, 10, 10)) < 0.5

    with pytest.raises(ValueError):
        mcubes.smooth(
            binary_levelset,
            method='constrained',
            max_iters=500,
            rel_tol=1e-4
        )


def test_wrong_method():

    with pytest.raises(ValueError):
        mcubes.smooth(
            np.zeros((10, 10), dtype=np.bool_),
            method='wrong',
            max_iters=500,
            rel_tol=1e-4
        )


def test_circle():

    x, y = np.mgrid[:100, :100]
    levelset = np.sqrt((x - 50)**2 + (y - 50)**2) - 25
    binary_levelset = levelset > 0

    smoothed_levelset = mcubes.smooth(
        binary_levelset,
        max_iters=500,
        rel_tol=1e-4
    )

    assert np.all(np.abs(smoothed_levelset - levelset) < 1)
    assert np.all((smoothed_levelset > 0) == binary_levelset)


# if __name__ == '__main__':
#     # logging.basicConfig(level=logging.DEBUG)
#     test_circle()
#     test_sphere()
#     test_large_sphere()
