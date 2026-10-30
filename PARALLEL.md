# Parallel marching cubes

`mcubes.marching_cubes` and `mcubes.marching_cubes_func` accept a `num_threads`
argument. The volume is swept in several passes — classify the cubes, prefix sum
their vertex counts, place the vertices, emit the triangles — and each pass runs
on all requested threads. The extracted mesh is identical whatever the thread
count.

```python
vertices, triangles = mcubes.marching_cubes(volume, isovalue, num_threads=4)
```

`num_threads=0` (the default) uses one thread per CPU core and `1` keeps the
whole computation on the calling thread.

| | old | new, 4 threads |
|---|---|---|
| binary 100³ (README `binary.jpg`) | 19.3 ms | **4.2 ms** (4.6×) |
| smoothed 100³ (README `smooth.jpg`) | 19.2 ms | **4.1 ms** (4.6×) |
| sphere 256³ | 310 ms | **80 ms** (3.9×) |

---

## 1. Background

### 1.1 Standard marching cubes

The grid has `numx × numy × numz` scalar samples, i.e. `nx × ny × nz` cells
(`nx = numx - 1`, …). Cube `(i, j, k)` has eight corners at the grid nodes
`(i + dx, j + dy, k + dz)` for `dx, dy, dz ∈ {0, 1}`, numbered so that bit `m` of
the cube index refers to corner `m`:

```
        v7 ---------- v6          edges:  0: v0-v1   1: v1-v2   2: v2-v3   3: v3-v0
       / |           /|                  4: v4-v5   5: v5-v6   6: v6-v7   7: v7-v4
     v4 ---------- v5 |                  8: v0-v4   9: v1-v5  10: v2-v6  11: v3-v7
      |  |          |  |
      | v3 ---------| v2                v0=(0,0,0) v1=(1,0,0) v2=(1,1,0) v3=(0,1,0)
      | /           | /                 v4=(0,0,1) v5=(1,0,1) v6=(1,1,1) v7=(0,1,1)
     v0 ---------- v1
```

`cube_index` marks the corners below the isovalue, `edge_table[cube_index]` marks
the edges crossing the surface, and `triangle_table[cube_index]` lists the
triangles as triples of edge ids. A vertex is placed on each crossing edge by
linear interpolation between its two corners.

### 1.2 Why the original implementation could not be parallelised

A naive implementation emits three vertices per triangle, which duplicates every
vertex six times and destroys the mesh. The original code avoided duplicates by
sweeping the volume in a fixed order and remembering, in three small index
buffers, which vertex had already been created for the edges shared with the
previously visited cubes:

```
v[4] = f(x, y, lower[2]); ...                      // 4 evaluations per row
for k:                                            // rolling reuse along z
    v[0] = v[4]; ...
    v[4] = f(x, y, z_dz); ...                     // 4 evaluations per cube
    ... edge 6: create vertex, write shared_indices_x[...]
    ... edge 0: if (j == 0 && k == 0) create, else read shared_indices_x[...]
```

Two things make this inherently sequential:

* the shared-index buffers are written by one cube and read by the next, so the
  visitation order *is* the algorithm; and
* vertices are appended to a `std::vector` with `push_back`, so the final vertex
  numbering follows the sweep order.

Every cube is also evaluated four times per grid node (the rolling window fetches
a whole new `z` layer per cube), which is why `marching_cubes_func` had to pay 4×
the Python callbacks, and the `push_back` growth dominates on dense fields.

---

## 2. Algorithm design

The parallel formulation replaces "create a vertex the first time an edge is
seen" with "**give every edge exactly one owning cube, then fill in the counts,
offsets and positions in independent passes**".

### 2.1 Edge ownership

Every cell edge lies along one axis and is shared by up to four cubes. Pick one
owner per edge line:

> **A cube owns the three edges that start at its minimum corner: edge 0 (`v0→v1`,
> +x), edge 3 (`v3→v0`, +y) and edge 8 (`v0→v4`, +z).**

For an interior cube these three edges are the `x` line at `(j, k)`, the `y` line
at `(i, k)` and the `z` line at `(i, j)`; the other nine edges belong to
neighbouring cubes at `+x`, `+y` and `+z`. Three properties follow:

* an edge crossing the surface is reported identically by all cubes that share
  it (same two corner values, hence the same `edge_table` bit), so the owner
  always sees the bit that a neighbour would have seen;
* the three owned edges of a cube are disjoint from those of every other cube,
  so no two cubes can create the same vertex; and
* all neighbour lookups go in the `+x`, `+y`, `+z` direction, so no negative
  indices are ever formed.

The rule has to hold for the edges on the upper boundary too. Those lie on the
maximum face of the last cube, which does not own them by the rule above.

### 2.2 The ghost layer

The fix is to extend the cube grid by one cell at the top-right boundary of every
axis. The domain becomes `(nx+1) × (ny+1) × (nz+1)` cubes, of which those with
`i == nx`, `j == ny` or `k == nz` are **ghosts**: they cover no volume, they own
edges, and they never emit a triangle.

```
                    i = 0      i = nx-1    i = nx
        j = ny    [ ghost ]   [ ghost ]   [ ghost ]   ghost row: owns the x lines
        ...       [ real  ]   [ real  ]   [ ghost ]   and z lines at j = ny
        j = 0     [ real  ]   [ real  ]   [ ghost ]   ghost column: owns the y
                                                    lines and z lines at i = nx
```

Now the ownership rule is uniform over the whole extended box: every edge of the
volume is the minimum-corner edge of exactly one cube, real or ghost. Ghost cubes
own only the edges that lie *on* the boundary plane, which is precisely what is
needed, and their invalid edges never fire because the corner reads clamp:

> A corner index that would leave the grid is clamped to the boundary node. The
> two endpoints of such an edge are then the same node, so both are below (or
> both above) the isovalue and `edge_table` reports no crossing for it.

That single convention removes every boundary special case: no masks, no
`if (j == 0)` branches, and no per-axis bookkeeping in the hot loop.

Edges are resolved to their owner with a fixed table (`owner edge` is the edge id
the owner uses, and the rank is its ordinal among the owner's vertices):

| edge | line | owner cube | owner edge |
| ---- | ---- | ---------- | ---------- |
| 0 | x at (j, k) | (i, j, k) | 0 |
| 1 | y at (i+1, k) | (i+1, j, k) | 3 |
| 2 | x at (j+1, k) | (i, j+1, k) | 0 |
| 3 | y at (i, k) | (i, j, k) | 3 |
| 4 | x at (j, k+1) | (i, j, k+1) | 0 |
| 5 | y at (i+1, k+1) | (i+1, j, k+1) | 3 |
| 6 | x at (j+1, k+1) | (i, j+1, k+1) | 0 |
| 7 | y at (i, k+1) | (i, j, k+1) | 3 |
| 8 | z at (i, j) | (i, j, k) | 8 |
| 9 | z at (i+1, j) | (i+1, j, k) | 8 |
| 10 | z at (i+1, j+1) | (i+1, j+1, k) | 8 |
| 11 | z at (i, j+1) | (i, j+1, k) | 8 |

Because each owner has at most three vertices, its rank is a couple of bit tests:
`rank(0) = 0`, `rank(3) = (edges & 1)`, `rank(8) = (edges & 1) + (edges & 8)`.

### 2.3 Why a ghost layer instead of boundary cases

The reference this work started from
([AIHoloImager `MarchingCubes.cpp`](https://github.com/gongminmin/AIHoloImager/blob/3186611474345c3f0d752752512fd4d02efec9a8/Source/Lib/Source/MeshGen/MarchingCubes.cpp))
uses the same three owned edges but patches the boundary with extra owned sets per
axis — `x == max` also owns `{1, 9}`, `y == max` owns `{2, 11}`, `z == max` owns
`{4, 7}`. Three edge classes (5, 6 and 10) are then never owned by any cube and are
resolved through diagonal-neighbour lookups whose `(x+1, …, z+1)` indices run past
the end of the flat arrays — or address an unrelated cube — when the cube already
sits on two maximum faces. The ghost layer makes those edges ordinary
minimum-corner edges, so the whole class of double-boundary special cases
disappears.

### 2.4 The five passes

`c(i, j, k) = ((i·(ny+1)) + j)·(nz+1) + k` addresses the extended grid.

```
pass 1  (parallel, per row of cubes along z)
        cube_index[c]     = bits of the 8 clamped corner values <= isovalue
        vertex_offset[c]  = number of set bits among {0, 3, 8} of edge_table
        polygon_offset[c] = 0 for ghosts, num_indices_table[cube_index] otherwise

pass 2  (parallel scan)
        vertex_offset  := exclusive prefix sum of vertex counts   -> V
        polygon_offset := exclusive prefix sum of polygon counts  -> F·3

pass 3  (parallel)   for every cube owning vertices, write its 1..3 vertices at
                     vertex_offset[c] + rank

pass 4  (parallel)   for every real cube, resolve its 12 edge vertices through
                     the owner table above and copy triangle_table[ci] scaled
                     by polygon_offset[c]
```

Pass 2 is what makes the later passes parallel: every cube gets a private output
range, so pass 3 and pass 4 never write the same location, and pass 4 can resolve
any edge because *all* vertices already exist.

### 2.5 Determinism

Output order is defined by the sweep, not by the schedule:

* vertices are numbered in extended-cube order (pass 2's scan is order-preserving);
* within a cube, vertices follow the fixed rank order 0, 3, 8;
* triangles are emitted in real-cube order, in `triangle_table` order.

Threads only partition ranges, and every pass is a pure function of the grid, so
1 thread and 24 threads produce byte-identical arrays. This is verified in
§4.2 and by `test_num_threads_array` / `test_num_threads_func`.

### 2.6 Cost model

Memory, transient unless stated otherwise:

| | size |
| --- | --- |
| per extended cube | 9 B — `u8` cube index + two `u32` offsets (9 MB at 100³, 153 MB at 256³) |
| per vertex | 24 B (3 × `double`, returned) |
| per polygon index | 8 B (`size_t`, freed once the NumPy array is filled) |

Work: pass 1 reads every grid node once (four reads per cube, reused along the
row) and stores three words; pass 3 re-reads four corner values for the cubes
that own a vertex; pass 4 touches no grid data at all. So the volume is streamed
twice and the total cost is `O(N + V + F)`.

The second stream is why the multi-pass version does more memory traffic than the
old single sweep — but it writes into pre-sized arrays instead of growing a
`std::vector`, which more than pays for it: the implementation below is 1.6–2.2×
faster than the old sweep even on a single thread (see §4.3).

---

## 3. Implementation

### 3.1 File map

| file | role |
| --- | --- |
| `mcubes/src/marchingcubes.h` | `thread_pool`, `exclusive_scan`, `marching_cubes_grid` (the five passes), `marching_cubes` entry point with `num_threads` |
| `mcubes/src/marchingcubes.cpp` | `edge_table`, `triangle_table`, `num_indices_table` (precalculated index counts per cube configuration), interpolation |
| `mcubes/src/pywrapper.cpp` | array/`func` entry points, dtype dispatch, GIL release, grid sampling for `marching_cubes_func` |
| `mcubes/src/_mcubes.pyx` | public `marching_cubes(volume, isovalue, num_threads=0)` |
| `setup.py` | `-pthread` on POSIX (needed to link `std::thread`) |

### 3.2 Thread pool

`private_::thread_pool` keeps `num_threads - 1` workers alive for the duration of
one call; the calling thread joins in, so a pool of one never spawns a thread.

`parallel_for(count, body)` splits `[0, count)` into grain-sized chunks published
through one `std::function`, and workers steal chunks with an atomic counter, so
an uneven pass still balances. Completion uses a generation counter plus a
pending-worker count; a throwing body is captured in an `exception_ptr` and
rethrown by `parallel_for` after all workers are idle, so an exception can never
deadlock the pool.

Threads are only created when the volume is worth it
(`(nx+1)(ny+1)(nz+1) >= 2^15`); below that the pool is built with one thread and
everything runs inline, so small volumes pay nothing for the machinery.

### 3.3 Parallel prefix sum

`exclusive_scan` is a two-level scan over `u32` counts, in place: each worker sums
a grain, the grain sums are scanned serially, then each worker writes its offsets
starting from its grain base. It falls back to a serial loop below 65 536
elements or with one thread, and raises `RuntimeError` if the total exceeds
`2^32` (the offset arrays are 32 bit).

### 3.4 Classification pass

The inner loop walks whole `z` rows so the four corners shared with the next cube
stay in registers, which keeps the grid traffic at ~1 read per node:

```cpp
double v4 = value_at(i, j, 0), v5 = value_at(i1, j, 0),
       v6 = value_at(i1, j1, 0), v7 = value_at(i, j1, 0);
for(size_t k=0; k<cz; ++k, ++cube) {
    const size_t k1 = (k < nz) ? k + 1 : nz;   // clamps on the ghost layer
    const double v0 = v4, v1 = v5, v2 = v6, v3 = v7;
    v4 = value_at(i, j, k1); ...
    unsigned ci = 0;
    if(v0 <= isovalue) ci |= 1;  ...  if(v7 <= isovalue) ci |= 128;
    cube_index[cube] = ci;
    vertex_offset[cube] = owned_edge_count(edge_table[ci]);
    polygon_offset[cube] = (has_x && has_y && has_z) ? num_indices_table[ci] : 0;
}
```

`i1`/`j1`/`k1` are clamped, which is what implements §2.2's "invalid edges are
degenerate" rule without any branch.

### 3.5 Vertex placement

Only three edges per cube can own a vertex, and only four corner values are
needed (`v0`, `v1`, `v3`, `v4`), so the pass re-reads those and interpolates:

```cpp
if(edges & 0x001) { *vertex++ = mc_isovalue_interpolation(iso, v0, v1, x_i,  x_i1);
                    *vertex++ = y_j; *vertex++ = z_k; }
if(edges & 0x008) { *vertex++ = x_i;
                    *vertex++ = mc_isovalue_interpolation(iso, v3, v0, y_j1, y_j); *vertex++ = z_k; }
if(edges & 0x100) { *vertex++ = x_i; *vertex++ = y_j;
                    *vertex++ = mc_isovalue_interpolation(iso, v0, v4, z_k, z_k1); }
```

Node coordinates are precomputed once (`x0 + dx * i`, the same expression the
original used) so the interpolation is evaluated on identical inputs.

### 3.6 Polygon emission

For each real cube with a non-empty `triangle_table` row, the 12 edge vertices are
resolved from the owner table of §2.2 and then copied through
`triangle_table[ci]` (`owner(c, e)` below is shorthand for
`vertex_offset[c] + owned_edge_rank(edge_table[cube_index[c]], e)`):

```cpp
index[0] = vertex_offset[cube];
index[3] = vertex_offset[cube] + owned_edge_rank(edges, 3);
index[8] = vertex_offset[cube] + owned_edge_rank(edges, 8);
if(edges & 0x002) index[1]  = owner(cube_i1, 3);        // +x neighbour
if(edges & 0x004) index[2]  = vertex_offset[cube_j1];   // owner edge 0
if(edges & 0x400) index[10] = owner(cube_i1 + cz, 8);   // +x +y diagonal
...
```

Note that `triangle_table` rows reference only edges whose `edge_table` bit is
set, so `index[]` is fully initialised for every cube that reaches the copy loop;
this invariant holds for all 256 rows.

### 3.7 Python layer

* `num_threads` is a keyword on both entry points, validated (`>= 0`).
* The GIL is released (RAII `gil_release`) around the whole computation.
* The array path dispatches the dtype **once** into a typed strided reader
  (`array_value_at<T>`), so `bool` masks, integer volumes and `float64` all read
  at the same speed, and negative strides / Fortran order work unchanged. Input
  that is byte-swapped or unaligned is first converted to a native, aligned
  `double` copy, because the reader dereferences the buffer directly.
* A grid whose element count would overflow `size_t` is rejected with a
  `RuntimeError` before anything is allocated.
* `marching_cubes_func` evaluates `f(x, y, z)` **once per grid node** into a
  `double` buffer — 4× fewer Python callbacks than the old rolling window — and
  then runs the same parallel core on that buffer. Sampling must stay on the
  calling thread because it calls back into Python. If `f` raises or returns a
  non-number, the call aborts and re-raises that exception instead of building a
  surface with the failing nodes replaced by `0.0`.

### 3.8 Numerical behaviour

Vertices are placed with the original formula
`(x2-x1)*(isovalue-f1)/(f2-f1) + x1`, but each owned edge is always interpolated
from its low corner, whereas the old code used the direction implied by the edge
table. The two are algebraically identical; they differ by at most 1 ULP when
floating point rounding lands differently. Integer and boolean volumes are
bit-identical (the weight is exactly `0.5`), and floating point volumes agree to
`≤ 3.6e-15` in absolute terms (§4.1).

---

## 4. Validation

Machine: Windows 11, 24 logical cores, MSVC 19.51, Python 3.11, NumPy 2.4.

### 4.1 Mesh equivalence

Compared against the previous implementation, order-independently (vertex
positions plus the set of triangles expressed by their corner coordinates):

| corpus | result |
| --- | --- |
| 200 random volumes (noise, integer fields, sinusoids, spheres, quantised) | 200/200 identical triangle soup |
| 16 dtype/layout cases (`bool`, all int widths, `float32/64`, Fortran order, strided, negative strides; `longdouble` is `float64` on Windows) | identical soup; max vertex deviation `0` for `bool`/integer, `3.55e-15` for `float32/64` |
| 10 fixed cases (empty, sphere, torus, noise, 2×2×2, boundary-touching) | identical soup and counts |

Topology is also asserted in the test suite: for a 40³ sphere `V - E + F = 2`, all
indices are in range, and no triangle is degenerate.

`python -m pytest -q` → 20 passed, `flake8 .` → clean.

### 4.2 Regenerated illustrations

`images/binary.jpg`, `images/smooth.jpg` and `images/smoothing_overview.png` were
rebuilt from the meshes of §5 with an orthographic z-buffer renderer using the
same framing, grey background and top lighting, once from the old build and once
from the new one:

| image | old vs new | new vs committed image |
| --- | --- | --- |
| `binary.jpg` | 0 of 65 536 pixels differ | silhouette IoU 0.91, grey correlation 0.67 |
| `smooth.jpg` | 9 of 65 536, each by 1 grey level | silhouette IoU 0.92, grey correlation 0.75 |
| `smoothing_overview.png` | 1 of 271 040, by 1 grey level | same 2×2 layout, panels and labels |

The handful of 1-level pixels come from §3.8: 53 of 11 646 vertices of the
smoothed mesh differ by ≤ 1.4e-14, which flips a depth tie on a shared edge where
the surface is saturating. The remaining difference to the committed images is
lighting and camera only — the originals were produced by another renderer (and
the overview was assembled by hand in Inkscape).

Rendering with 1, 2, 4 and 8 threads produces byte-identical images.

### 4.3 Performance

Best of 2–7 runs, timing the `marching_cubes` call only. Run-to-run variation on
this machine reaches 7–20 % for the 100³ cases, so treat differences below ~10 %
as noise.

| workload | old (1 t) | new 1 t | new 2 t | **new 4 t** | new 8 t | new auto (24 t) | **4 t vs old** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binary 100³ (`binary.jpg`) | 19.33 | 9.31 | 6.18 | **4.17** | 3.44 | 3.73 | **4.64×** |
| smoothed 100³ (`smooth.jpg`) | 19.21 | 8.56 | 5.75 | **4.14** | 3.22 | 3.78 | **4.64×** |
| sphere 128³ (`float64`) | 34.51 | 19.46 | 12.92 | **8.83** | 7.96 | 8.49 | **3.91×** |
| sphere 256³ (`int64`) | 310.11 | 153.33 | 107.83 | **80.24** | 73.22 | 72.31 | **3.86×** |
| dense noise 160³ | 564.69 | 341.38 | 261.72 | **228.49** | 223.35 | 223.37 | **2.47×** |
| `marching_cubes_func` 100³ | 947.08 | 249.92 | 248.74 | **257.03** | 254.33 | 262.72 | **3.68×** |

All times in milliseconds.

Reading the table:

* **Threads help most for sparse surfaces** (`4.6×` on both 100³ cases), which are
  dominated by the grid reads in passes 1 and 3.
* **Dense fields saturate memory bandwidth** — the noise volume generates 6.1 M
  vertices and 12.9 M indices, so 4 threads give only `2.5×` and the curve is flat
  from 8 threads on.
* **The `func` API barely benefits from threads**: the GIL-bound Python callbacks
  in the sampling loop dominate and run on one thread, so its 1–24 thread columns
  are all within noise. Its `3.7×` win over the old implementation comes from
  evaluating each grid node once instead of four times.
* **One thread is already faster than the old sweep everywhere** (1.65–2.24×, and
  3.8× for `func`): the passes cost an extra stream over the volume but remove the
  `push_back` growth and the shared-index bookkeeping.
* Auto (24 threads) is not the best choice for memory-bound volumes on a large
  machine; it helps sparse surfaces slightly and is within noise elsewhere.

---

## 5. Reproducing

```bash
python -m pip install -e .
python -m pytest -q
```

The meshes behind the illustrations:

```python
import numpy as np, mcubes

x, y, z = np.mgrid[:100, :100, :100]
binary_sphere = (x - 50)**2 + (y - 50)**2 + (z - 50)**2 - 25**2 < 0
vertices, triangles = mcubes.marching_cubes(binary_sphere, 0.5, num_threads=4)

smoothed_sphere = mcubes.smooth(binary_sphere)
vertices, triangles = mcubes.marching_cubes(smoothed_sphere, 0, num_threads=4)
```

A timing harness for any volume and thread count:

```python
import time, numpy as np, mcubes

x, y, z = np.mgrid[:256, :256, :256]
volume = (x - 128)**2 + (y - 128)**2 + (z - 128)**2 - 96**2

for num_threads in (1, 2, 4, 8, 0):
    times = []
    for _ in range(5):
        start = time.perf_counter()
        mcubes.marching_cubes(volume, 0.0, num_threads=num_threads)
        times.append(time.perf_counter() - start)
    print(f"{num_threads:2d} threads: {min(times) * 1000:6.1f} ms")
```

---

## 6. Notes and limitations

* **Scratch memory** is 9 bytes per extended cube — ~9 MB for 100³ and ~153 MB for
  256³ — on top of the vertices (24 B each) and the polygon buffer (8 B per index
  while the call runs). The arrays are allocated whatever the thread count, so a
  volume that only just fits in memory today may need a smaller grid here.
* **32-bit offsets.** Vertex and polygon counts above `2^32` raise
  `RuntimeError: Mesh is too large for 32 bit indices.` A 4096³ smooth surface
  produces roughly 5·10⁸ indices, still an order of magnitude below the limit.
* **Small volumes stay serial** below 32 768 cubes; `num_threads` has no effect
  there and no threads are created.
* **Nested parallelism.** Each call builds its own pool, so calling
  `marching_cubes` from inside another thread pool multiplies the thread count;
  pass a smaller `num_threads` in that case.
* **The C++ entry point changed contract.** `mc::marching_cubes` now takes a
  `value_at(i, j, k)` callable that receives **grid node indices**, not physical
  coordinates, and it may be called from several threads at once when
  `num_threads > 1`. A lambda written against the old coordinate-based signature
  still compiles but would return the wrong field, so C++ callers that pass their
  own callable must adapt it. The Python API is unaffected:
  `mcubes.marching_cubes_func` still calls `f(x, y, z)` with coordinates. The
  lookup tables (`edge_table`, `triangle_table`, `num_indices_table`) are now
  declared `const`, so code that wrote to them no longer compiles.
* The polygon buffer is still returned as `NPY_ULONG`, matching the previous
  behaviour (32-bit indices on Windows, 64-bit elsewhere).
