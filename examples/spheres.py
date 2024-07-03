
import numpy as np
import mcubes

print("Example 1: Isosurface in NumPy volume...")

# Create a data volume
X, Y, Z = np.mgrid[:100, :100, :100]
u = (X-50)**2 + (Y-50)**2 + (Z-50)**2 - 25**2

# Extract the 0-isosurface
vertices1, triangles1 = mcubes.marching_cubes(u, 0)

# Export the result to sphere.dae
mcubes.export_mesh(vertices1, triangles1, "sphere1.dae", "MySphere")

print("Done. Result saved in 'sphere1.dae'.")

print("Example 2: Isosurface in Python function...")
print("(this might take a while...)")


# Create the volume
def f(x, y, z):
    return x**2 + y**2 + z**2


# Extract the 16-isosurface
vertices2, triangles2 = mcubes.marching_cubes_func(
        (-10, -10, -10),    # Lower bound
        (10, 10, 10),       # Upper bound
        100, 100, 100,      # Number of samples in each dimension
        f,                  # Implicit function
        16)                 # Isosurface value

# Export the result to sphere2.dae
mcubes.export_mesh(vertices2, triangles2, "sphere2.dae", "MySphere")
print("Done. Result saved in 'sphere2.dae'.")

try:
    print("Plotting mesh...")
    from mayavi import mlab
    mlab.triangular_mesh(
        vertices1[:, 0], vertices1[:, 1], vertices1[:, 2],
        triangles1)
    print("Done.")
    mlab.show()
except ImportError:
    print("Could not import mayavi. Interactive demo not available.")
