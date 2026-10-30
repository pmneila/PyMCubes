
import pytest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

import mcubes


def test_empty():

    levelset = np.zeros((50, 50, 50))
    vertices, triangles = mcubes.marching_cubes(levelset, 0.5)

    assert len(vertices) == len(triangles) == 0


def test_sphere():
    x, y, z = np.mgrid[:100, :100, :100]
    u = (x - 50)**2 + (y - 50)**2 + (z - 50)**2 - 25**2

    def func(x, y, z):
        return (x - 50)**2 + (y - 50)**2 + (z - 50)**2 - 25**2

    vertices1, triangles1 = mcubes.marching_cubes(u, 0.0)
    vertices2, triangles2 = mcubes.marching_cubes_func(
        (0, 0, 0),
        (99, 99, 99),
        100, 100, 100,
        func, 0.0
    )

    assert_allclose(vertices1, vertices2)
    assert_array_equal(triangles1, triangles2)


def test_no_duplicates():
    def sphere(x, y, z):
        return np.sqrt((x - 4)**2 + (y - 4)**2 + (z - 4)**2) - 4

    vertices, _ = mcubes.marching_cubes_func((2, 2, 2), (9, 9, 9), 20, 20, 20, sphere, 0)

    assert len(vertices) == len(np.unique(vertices, axis=0))


def test_export():

    u = np.zeros((10, 10, 10))
    u[2:-2, 2:-2, 2:-2] = 1.0
    vertices, triangles = mcubes.marching_cubes(u, 0.5)

    mcubes.export_obj(vertices, triangles, "output/test.obj")
    mcubes.export_off(vertices, triangles, "output/test.off")
    mcubes.export_mesh(vertices, triangles, "output/test.dae")


def test_invalid_input():

    def func(x, y, z):
        return x**2 + y**2 + z**2 - 1

    mcubes.marching_cubes_func((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), 10, 10, 10, func, 0)

    with pytest.raises(ValueError):
        mcubes.marching_cubes_func((0, 0, 0), (0, 0, 0), 10, 10, 10, func, 0)

    with pytest.raises(ValueError):
        mcubes.marching_cubes_func((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), 1, 10, 10, func, 0)

    with pytest.raises(Exception):
        mcubes.marching_cubes_func((-1.5, -1.5), (1.5, 1.5, 1.5), 10, 10, 10, func, 0)

    with pytest.raises(Exception):
        mcubes.marching_cubes_func((-1.5, -1.5, -1.5), (1.5, 1.5), 10, 10, 10, func, 0)


def test_num_threads_invalid():

    levelset = np.zeros((10, 10, 10))

    with pytest.raises(ValueError):
        mcubes.marching_cubes(levelset, 0.5, num_threads=-1)

    def func(x, y, z):
        return x**2 + y**2 + z**2 - 1

    with pytest.raises(ValueError):
        mcubes.marching_cubes_func(
            (-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), 10, 10, 10, func, 0, num_threads=-1
        )


@pytest.mark.parametrize("num_threads", [1, 2, 3, 8])
def test_num_threads_array(num_threads):
    # A volume large enough to take the parallel path.
    x, y, z = np.mgrid[:60, :52, :44]
    volume = (x - 30) ** 2 + (y - 26) ** 2 + (z - 22) ** 2 - 18**2

    vertices, triangles = mcubes.marching_cubes(volume, 0.0, num_threads=1)
    vertices_threaded, triangles_threaded = mcubes.marching_cubes(
        volume, 0.0, num_threads=num_threads
    )

    assert len(vertices) > 0
    assert_array_equal(vertices, vertices_threaded)
    assert_array_equal(triangles, triangles_threaded)


@pytest.mark.parametrize("num_threads", [1, 2, 3, 8])
def test_num_threads_func(num_threads):
    def func(x, y, z):
        return (x - 6)**2 + (y - 6)**2 + (z - 6)**2 - 25

    vertices, triangles = mcubes.marching_cubes_func(
        (0, 0, 0), (12, 12, 12), 40, 40, 40, func, 0.0, num_threads=1
    )
    vertices_threaded, triangles_threaded = mcubes.marching_cubes_func(
        (0, 0, 0), (12, 12, 12), 40, 40, 40, func, 0.0, num_threads=num_threads
    )

    assert len(vertices) > 0
    assert_array_equal(vertices, vertices_threaded)
    assert_array_equal(triangles, triangles_threaded)


def test_closed_sphere():
    # Every edge of a closed surface is shared by exactly two triangles, so
    # V - E + F must be 2 for a sphere and 0 for a torus. This catches any
    # duplicated or dropped vertex.
    x, y, z = np.mgrid[:40, :40, :40]
    sphere = (x - 20) ** 2 + (y - 20) ** 2 + (z - 20) ** 2 - 12**2
    vertices, triangles = mcubes.marching_cubes(sphere, 0.0, num_threads=4)

    edges = np.sort(
        np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]),
        axis=1,
    )
    num_edges = len(np.unique(edges, axis=0))

    assert triangles.min() >= 0
    assert triangles.max() < len(vertices)
    assert len(vertices) - num_edges + len(triangles) == 2


class CallbackError(Exception):
    pass


def test_func_exception_propagates():
    # A failing callback must abort with its own exception, not silently produce
    # a surface with the failing nodes replaced by 0.0.
    def func(x, y, z):
        if x > 1:
            raise CallbackError("callback failed")
        return 0.0

    with pytest.raises(CallbackError):
        mcubes.marching_cubes_func((0, 0, 0), (2, 2, 2), 5, 5, 5, func, 0.0)


def test_func_bad_return_value():
    def func(x, y, z):
        return "not a number"

    with pytest.raises(TypeError):
        mcubes.marching_cubes_func((0, 0, 0), (2, 2, 2), 5, 5, 5, func, 0.0)


def test_func_grid_overflow():
    with pytest.raises(RuntimeError):
        mcubes.marching_cubes_func(
            (0, 0, 0), (1, 1, 1), 1 << 22, 1 << 22, 1 << 22, lambda x, y, z: 0.0, 0.0
        )


def test_non_native_byte_order():
    x, y, z = np.mgrid[:12, :12, :12]
    volume = (x - 6)**2 + (y - 6)**2 + (z - 6)**2 - 4**2
    ref_vertices, ref_triangles = mcubes.marching_cubes(volume, 0.0, num_threads=1)

    swapped = volume.astype(volume.dtype.newbyteorder())
    assert not swapped.dtype.isnative

    vertices, triangles = mcubes.marching_cubes(swapped, 0.0, num_threads=2)
    assert_allclose(vertices, ref_vertices)
    assert_array_equal(triangles, ref_triangles)


def test_unaligned_input():
    x, y, z = np.mgrid[:12, :12, :12]
    volume = ((x - 6)**2 + (y - 6)**2 + (z - 6)**2 - 4**2).astype(np.float64)
    ref_vertices, ref_triangles = mcubes.marching_cubes(volume, 0.0, num_threads=1)

    raw = np.zeros(volume.nbytes + 8, dtype=np.uint8)
    unaligned = raw[1:1 + volume.nbytes].view(np.float64).reshape(volume.shape)
    unaligned[...] = volume
    assert not unaligned.flags["ALIGNED"]

    vertices, triangles = mcubes.marching_cubes(unaligned, 0.0, num_threads=2)
    assert_allclose(vertices, ref_vertices)
    assert_array_equal(triangles, ref_triangles)


def test_dtype_support_is_layout_independent():
    # Every dtype numpy can safely cast must be accepted, whether or not the
    # array happens to be native-endian and aligned.
    x, y, z = np.mgrid[:10, :10, :10]
    field = (x - 5)**2 + (y - 5)**2 + (z - 5)**2 - 3**2
    ref_vertices, ref_triangles = mcubes.marching_cubes(field.astype(np.float64), 0.0, num_threads=1)

    # float16 has 11 mantissa bits, so these integer values are exact.
    half = field.astype(np.float16)
    for volume in (half, half.astype(half.dtype.newbyteorder())):
        vertices, triangles = mcubes.marching_cubes(volume, 0.0, num_threads=2)
        assert_allclose(vertices, ref_vertices)
        assert_array_equal(triangles, ref_triangles)

    complex_volume = np.zeros((4, 4, 4), dtype=np.complex128)
    for volume in (complex_volume, complex_volume.astype(complex_volume.dtype.newbyteorder())):
        with pytest.raises(TypeError):
            mcubes.marching_cubes(volume, 0.0)


def test_every_cube_configuration():
    # Exercises all 256 entries of num_indices_table through triangle_table. A
    # count that is too large would make the polygon copy loop step past the -1
    # terminator and emit an out-of-range or repeated index.
    #
    # Cubes number their corners around each z face, not as a bit mask.
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

    for cube_index in range(256):
        volume = np.empty((2, 2, 2))
        for corner, (di, dj, dk) in enumerate(corners):
            # 0.0 is at or below the isovalue, so the corner is inside when its
            # bit is set and the cube index equals cube_index.
            volume[di, dj, dk] = 0.0 if (cube_index >> corner) & 1 else 1.0

        vertices, triangles = mcubes.marching_cubes(volume, 0.5, num_threads=1)

        assert len(vertices) <= 12, cube_index
        if len(triangles) == 0:
            continue
        assert triangles.min() >= 0, cube_index
        assert triangles.max() < len(vertices), cube_index
        assert np.all(triangles[:, 0] != triangles[:, 1]), cube_index
        assert np.all(triangles[:, 1] != triangles[:, 2]), cube_index
        assert np.all(triangles[:, 0] != triangles[:, 2]), cube_index
