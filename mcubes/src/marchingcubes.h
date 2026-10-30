#ifndef _MARCHING_CUBES_H
#define _MARCHING_CUBES_H

#include <stddef.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace mc
{

extern const int edge_table[256];
extern const int triangle_table[256][16];
extern const int num_indices_table[256];

namespace private_
{

double mc_isovalue_interpolation(double isovalue, double f1, double f2,
    double x1, double x2);

inline int default_num_threads()
{
    static const int num = []() -> int
    {
        const unsigned int hardware = std::thread::hardware_concurrency();
        return hardware == 0 ? 1 : static_cast<int>(hardware);
    }();
    return num;
}

// Minimal fixed size thread pool.  The thread creating the pool takes part in
// every parallel loop, so a pool of one runs everything serially without
// spawning any thread at all.
class thread_pool
{
public:
    explicit thread_pool(int num_threads)
        : next_(0), num_pending_(0), generation_(0), stop_(false)
    {
        int requested = num_threads > 0 ? num_threads : default_num_threads();
        requested = std::min(requested, 512);
        const size_t num_workers = requested > 1 ? static_cast<size_t>(requested - 1) : 0;
        workers_.reserve(num_workers);
        try
        {
            for(size_t i=0; i<num_workers; ++i)
                workers_.push_back(std::thread(&thread_pool::worker_loop, this));
        }
        catch(...)
        {
            // The destructor will not run for an object whose constructor threw,
            // so stop and join here: destroying a joinable std::thread calls
            // std::terminate.
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
            }
            wake_.notify_all();
            for(size_t i=0; i<workers_.size(); ++i)
                workers_[i].join();
            throw;
        }
    }

    ~thread_pool()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        wake_.notify_all();
        for(size_t i=0; i<workers_.size(); ++i)
            workers_[i].join();
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;

    int size() const { return static_cast<int>(workers_.size()) + 1; }

    // Calls body(first, last) on disjoint sub-ranges covering [0, count).
    template<typename body_type>
    void parallel_for(size_t count, const body_type& body)
    {
        if(count == 0)
            return;

        const size_t total = static_cast<size_t>(size());
        if(total <= 1)
        {
            body(static_cast<size_t>(0), count);
            return;
        }

        // A few chunks per thread, so that the atomic counter can rebalance a
        // pass whose rows have very different costs.
        const size_t num_parts = std::min(total * 4, count);
        const size_t grain = (count + num_parts - 1) / num_parts;
        const size_t num_grains = (count + grain - 1) / grain;

        // A throwing body must not leave the other threads waiting forever.
        std::exception_ptr error;
        std::mutex error_mutex;

        next_ = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            job_ = [&]()
            {
                for(;;)
                {
                    const size_t g = next_.fetch_add(1, std::memory_order_relaxed);
                    if(g >= num_grains)
                        return;
                    const size_t begin = g * grain;
                    const size_t end = std::min(count, begin + grain);
                    try
                    {
                        body(begin, end);
                    }
                    catch(...)
                    {
                        std::lock_guard<std::mutex> elock(error_mutex);
                        if(!error)
                            error = std::current_exception();
                        return;
                    }
                }
            };
            num_pending_ = workers_.size();
            ++generation_;
        }
        wake_.notify_all();
        job_();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [this]() { return num_pending_ == 0; });
            job_ = nullptr;
        }

        if(error)
            std::rethrow_exception(error);
    }

private:
    void worker_loop()
    {
        std::uint64_t seen = 0;
        for(;;)
        {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this, seen]() { return stop_ || generation_ != seen; });
                if(stop_)
                    return;
                seen = generation_;
                try
                {
                    job = job_;
                }
                catch(...)
                {
                    // The copy may allocate; nothing to run, but the decrement
                    // below must still happen or the caller waits forever.
                    job = nullptr;
                }
            }
            if(job)
                job();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --num_pending_;
                if(num_pending_ == 0)
                    done_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    std::function<void()> job_;
    std::atomic<size_t> next_;
    size_t num_pending_;
    std::uint64_t generation_;
    bool stop_;
};

// Turns per element counts into exclusive prefix offsets, in place, and returns
// the total.
inline std::uint32_t exclusive_scan(std::uint32_t* data, size_t count,
    thread_pool& pool)
{
    if(count == 0)
        return 0;

    const std::uint64_t max_value = 0xFFFFFFFFull;

    if(pool.size() <= 1 || count < (1u << 16))
    {
        std::uint64_t sum = 0;
        for(size_t i=0; i<count; ++i)
        {
            const std::uint32_t value = data[i];
            data[i] = static_cast<std::uint32_t>(sum);
            sum += value;
        }
        if(sum > max_value)
            throw std::runtime_error("Mesh is too large for 32 bit indices.");
        return static_cast<std::uint32_t>(sum);
    }

    const size_t num_grains = static_cast<size_t>(pool.size()) * 4;
    const size_t grain = (count + num_grains - 1) / num_grains;
    const size_t real_grains = (count + grain - 1) / grain;
    std::vector<std::uint64_t> grain_sums(real_grains);

    pool.parallel_for(real_grains, [&](size_t first, size_t last)
    {
        for(size_t g=first; g<last; ++g)
        {
            std::uint64_t sum = 0;
            const size_t stop = std::min(count, (g + 1) * grain);
            for(size_t i=g*grain; i<stop; ++i)
                sum += data[i];
            grain_sums[g] = sum;
        }
    });

    std::uint64_t total = 0;
    for(size_t g=0; g<real_grains; ++g)
    {
        const std::uint64_t sum = grain_sums[g];
        grain_sums[g] = total;
        total += sum;
    }
    if(total > max_value)
        throw std::runtime_error("Mesh is too large for 32 bit indices.");

    pool.parallel_for(real_grains, [&](size_t first, size_t last)
    {
        for(size_t g=first; g<last; ++g)
        {
            std::uint32_t sum = static_cast<std::uint32_t>(grain_sums[g]);
            const size_t stop = std::min(count, (g + 1) * grain);
            for(size_t i=g*grain; i<stop; ++i)
            {
                const std::uint32_t value = data[i];
                data[i] = sum;
                sum += value;
            }
        }
    });

    return static_cast<std::uint32_t>(total);
}

// A cube owns the three edges that start at its minimum corner.  Every edge of
// the volume is such an edge of exactly one cube, so every crossing yields
// exactly one vertex and the passes below are independent of each other.
inline std::uint32_t owned_edge_count(int edges)
{
    return static_cast<std::uint32_t>(((edges & 0x001) ? 1 : 0)
        + ((edges & 0x008) ? 1 : 0) + ((edges & 0x100) ? 1 : 0));
}

inline std::uint32_t owned_edge_rank(int edges, int edge)
{
    switch(edge)
    {
    case 0:
        return 0;
    case 3:
        return (edges & 0x001) ? 1u : 0u;
    case 8:
        return ((edges & 0x001) ? 1u : 0u) + ((edges & 0x008) ? 1u : 0u);
    default:
        break;
    }
    assert(false && "owned_edge_rank supports only edges 0, 3 and 8");
    return 0;
}

// Count-then-fill marching cubes over a regular grid.
//
// value_at(i, j, k) returns the scalar field at grid node (i, j, k) and must be
// callable concurrently.  The grid holds numx * numy * numz nodes, spanning
// [x0, x0 + dx * (numx - 1)] and likewise for the other axes.
//
// The volume is swept in passes that are individually parallel:
//   1. classify every cube and count the vertices it owns,
//   2. prefix sum those counts into vertex offsets,
//   3. compute every vertex position,
//   4. prefix sum the polygon counts into polygon offsets,
//   5. emit the polygons.
template<typename value_at_type>
void marching_cubes_grid(const value_at_type& value_at,
    size_t numx, size_t numy, size_t numz,
    double x0, double y0, double z0,
    double dx, double dy, double dz,
    double isovalue, int num_threads,
    std::vector<double>& vertices, std::vector<size_t>& polygons)
{
    if(numx < 2 || numy < 2 || numz < 2)
        return;

    const size_t nx = numx - 1;
    const size_t ny = numy - 1;
    const size_t nz = numz - 1;

    // One extra layer of "ghost" cubes past the upper boundary of each axis
    // owns the edges lying on that boundary.  Ghost cubes create vertices but
    // never polygons, so the ownership rule above holds for the whole extended
    // box and no axis needs a special case.
    const size_t cx = nx + 1;
    const size_t cy = ny + 1;
    const size_t cz = nz + 1;
    const size_t num_cubes = cx * cy * cz;
    const size_t num_rows = cx * cy;

    std::vector<double> xs(numx), ys(numy), zs(numz);
    for(size_t i=0; i<numx; ++i)
        xs[i] = x0 + dx * static_cast<double>(i);
    for(size_t j=0; j<numy; ++j)
        ys[j] = y0 + dy * static_cast<double>(j);
    for(size_t k=0; k<numz; ++k)
        zs[k] = z0 + dz * static_cast<double>(k);

    // Marching cubes is memory bound; spawning threads for a small volume costs
    // more than it saves.
    const size_t parallel_threshold = 1u << 15;
    thread_pool pool(num_cubes < parallel_threshold ? 1 : num_threads);

    std::vector<std::uint8_t> cube_index(num_cubes);
    std::vector<std::uint32_t> vertex_offset(num_cubes);
    std::vector<std::uint32_t> polygon_offset(num_cubes);

    // Pass 1: classify cubes and count the vertices they own.
    pool.parallel_for(num_rows, [&](size_t row_first, size_t row_last)
    {
        for(size_t row=row_first; row<row_last; ++row)
        {
            const size_t i = row / cy;
            const size_t j = row % cy;
            const bool has_x = i < nx;
            const bool has_y = j < ny;
            // Corners past the boundary read the boundary node again.  Such a
            // degenerate corner never reports a crossing, so ghost cubes need
            // no extra handling here.
            const size_t i1 = has_x ? i + 1 : nx;
            const size_t j1 = has_y ? j + 1 : ny;

            double v4 = value_at(i, j, 0);
            double v5 = value_at(i1, j, 0);
            double v6 = value_at(i1, j1, 0);
            double v7 = value_at(i, j1, 0);

            size_t cube = row * cz;
            for(size_t k=0; k<cz; ++k, ++cube)
            {
                const bool has_z = k < nz;
                const size_t k1 = has_z ? k + 1 : nz;

                const double v0 = v4;
                const double v1 = v5;
                const double v2 = v6;
                const double v3 = v7;
                v4 = value_at(i, j, k1);
                v5 = value_at(i1, j, k1);
                v6 = value_at(i1, j1, k1);
                v7 = value_at(i, j1, k1);

                unsigned int ci = 0;
                if(v0 <= isovalue) ci |= 1;
                if(v1 <= isovalue) ci |= 2;
                if(v2 <= isovalue) ci |= 4;
                if(v3 <= isovalue) ci |= 8;
                if(v4 <= isovalue) ci |= 16;
                if(v5 <= isovalue) ci |= 32;
                if(v6 <= isovalue) ci |= 64;
                if(v7 <= isovalue) ci |= 128;

                cube_index[cube] = static_cast<std::uint8_t>(ci);
                vertex_offset[cube] = owned_edge_count(edge_table[ci]);
                polygon_offset[cube] = (has_x && has_y && has_z)
                    ? static_cast<std::uint32_t>(num_indices_table[ci]) : 0;
            }
        }
    });

    // Pass 2: turn the per cube counts into offsets.
    const std::uint32_t num_vertices = exclusive_scan(
        vertex_offset.data(), num_cubes, pool);
    const std::uint32_t num_polygon_indices = exclusive_scan(
        polygon_offset.data(), num_cubes, pool);

    vertices.resize(static_cast<size_t>(num_vertices) * 3);
    polygons.resize(static_cast<size_t>(num_polygon_indices));

    // Pass 3: compute the position of every vertex.
    pool.parallel_for(num_rows, [&](size_t row_first, size_t row_last)
    {
        for(size_t row=row_first; row<row_last; ++row)
        {
            const size_t i = row / cy;
            const size_t j = row % cy;
            const size_t i1 = (i < nx) ? i + 1 : nx;
            const size_t j1 = (j < ny) ? j + 1 : ny;
            const double xi = xs[i];
            const double xi1 = xs[i1];
            const double yj = ys[j];
            const double yj1 = ys[j1];

            size_t cube = row * cz;
            for(size_t k=0; k<cz; ++k, ++cube)
            {
                const int edges = edge_table[cube_index[cube]];
                if(!owned_edge_count(edges))
                    continue;

                const size_t k1 = (k < nz) ? k + 1 : nz;
                const double v0 = value_at(i, j, k);
                double* vertex = &vertices[
                    static_cast<size_t>(vertex_offset[cube]) * 3];

                if(edges & 0x001)
                {
                    const double v1 = value_at(i1, j, k);
                    *vertex++ = mc_isovalue_interpolation(isovalue, v0, v1, xi, xi1);
                    *vertex++ = yj;
                    *vertex++ = zs[k];
                }
                if(edges & 0x008)
                {
                    const double v3 = value_at(i, j1, k);
                    *vertex++ = xi;
                    *vertex++ = mc_isovalue_interpolation(isovalue, v3, v0, yj1, yj);
                    *vertex++ = zs[k];
                }
                if(edges & 0x100)
                {
                    const double v4 = value_at(i, j, k1);
                    *vertex++ = xi;
                    *vertex++ = yj;
                    *vertex++ = mc_isovalue_interpolation(isovalue, v0, v4, zs[k], zs[k1]);
                }
            }
        }
    });

    // Pass 4: emit polygons, resolving the vertex of every shared edge through
    // the cube that owns it.
    pool.parallel_for(num_rows, [&](size_t row_first, size_t row_last)
    {
        for(size_t row=row_first; row<row_last; ++row)
        {
            const size_t i = row / cy;
            const size_t j = row % cy;
            if(i >= nx || j >= ny)
                continue;

            size_t cube = row * cz;
            for(size_t k=0; k<nz; ++k, ++cube)
            {
                const unsigned int ci = cube_index[cube];
                const size_t num_indices = static_cast<size_t>(num_indices_table[ci]);
                if(num_indices == 0)
                    continue;

                const int edges = edge_table[ci];
                const size_t cube_j1 = cube + cz;
                const size_t cube_k1 = cube + 1;
                const size_t cube_i1 = cube + cy * cz;

                std::uint32_t index[12] = {};
                index[0] = vertex_offset[cube];
                index[3] = vertex_offset[cube] + owned_edge_rank(edges, 3);
                index[8] = vertex_offset[cube] + owned_edge_rank(edges, 8);

                // x edges, owned by the +y and +z neighbours
                if(edges & 0x004) index[2] = vertex_offset[cube_j1];
                if(edges & 0x010) index[4] = vertex_offset[cube_k1];
                if(edges & 0x040) index[6] = vertex_offset[cube_j1 + 1];
                // y edges, owned by the +x and +z neighbours
                if(edges & 0x002)
                    index[1] = vertex_offset[cube_i1]
                        + owned_edge_rank(edge_table[cube_index[cube_i1]], 3);
                if(edges & 0x020)
                    index[5] = vertex_offset[cube_i1 + 1]
                        + owned_edge_rank(edge_table[cube_index[cube_i1 + 1]], 3);
                if(edges & 0x080)
                    index[7] = vertex_offset[cube_k1]
                        + owned_edge_rank(edge_table[cube_index[cube_k1]], 3);
                // z edges, owned by the +x and +y neighbours
                if(edges & 0x200)
                    index[9] = vertex_offset[cube_i1]
                        + owned_edge_rank(edge_table[cube_index[cube_i1]], 8);
                if(edges & 0x400)
                    index[10] = vertex_offset[cube_i1 + cz]
                        + owned_edge_rank(edge_table[cube_index[cube_i1 + cz]], 8);
                if(edges & 0x800)
                    index[11] = vertex_offset[cube_j1]
                        + owned_edge_rank(edge_table[cube_index[cube_j1]], 8);

                const int* triangles = triangle_table[ci];
                size_t* polygon = &polygons[polygon_offset[cube]];
                for(size_t m=0; m<num_indices; m+=3)
                {
                    polygon[m + 0] = index[triangles[m + 0]];
                    polygon[m + 1] = index[triangles[m + 1]];
                    polygon[m + 2] = index[triangles[m + 2]];
                }
            }
        }
    });
}

} // namespace private_

// Extracts the isovalue surface of a regular scalar grid.
//
// value_at(i, j, k) must return the field value at grid node (i, j, k), i.e. it
// receives integer node indices in [0, numx) x [0, numy) x [0, numz); it does
// NOT receive physical coordinates.  It is called once per node plus once more
// per cube that owns a vertex, from up to num_threads threads at the same time,
// so it must be thread-safe (with num_threads <= 1 it is only called from the
// calling thread).  num_threads = 0 means one thread per CPU core.
//
// The grid spans [lower, upper] and must hold numx * numy * numz >= 8 nodes.
// Vertices are emitted in (i, j, k) cube order; polygons index into them.
//
// Memory: besides the output mesh (24 bytes per vertex, sizeof(size) per polygon
// index), the algorithm allocates 9 bytes per (numx * numy * numz) cube of its
// extended grid, e.g. ~9 MB for a 100^3 node grid and ~153 MB for 256^3.  These
// buffers are allocated whatever num_threads is.
template<typename vector3, typename value_at_type>
void marching_cubes(const vector3& lower, const vector3& upper,
    int numx, int numy, int numz, value_at_type value_at, double isovalue,
    std::vector<double>& vertices, std::vector<typename vector3::size_type>& polygons,
    int num_threads = 0)
{
    using coord_type = typename vector3::value_type;

    // Some initial checks
    if(numx < 2 || numy < 2 || numz < 2)
        return;

    if(!std::equal(std::begin(lower), std::end(lower), std::begin(upper),
                   [](coord_type a, coord_type b)->bool {return a <= b;}))
        return;

    // numx, numy and numz are the numbers of evaluations in each direction
    const int num_cells_x = numx - 1;
    const int num_cells_y = numy - 1;
    const int num_cells_z = numz - 1;

    const double x0 = static_cast<double>(lower[0]);
    const double y0 = static_cast<double>(lower[1]);
    const double z0 = static_cast<double>(lower[2]);
    const double dx = static_cast<double>(
        (upper[0] - lower[0]) / static_cast<coord_type>(num_cells_x));
    const double dy = static_cast<double>(
        (upper[1] - lower[1]) / static_cast<coord_type>(num_cells_y));
    const double dz = static_cast<double>(
        (upper[2] - lower[2]) / static_cast<coord_type>(num_cells_z));

    private_::marching_cubes_grid(value_at,
        static_cast<size_t>(numx), static_cast<size_t>(numy), static_cast<size_t>(numz),
        x0, y0, z0, dx, dy, dz, isovalue, num_threads, vertices, polygons);
}

}

#endif // _MARCHING_CUBES_H
