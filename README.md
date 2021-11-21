# cuMesh
cuMesh is a simple C++/Cuda(thrust) parallel geometry processing library.

We develop it as a toolbox for post-processing of Delaunay 3d-reconstruction.


### Examples
- **Mesh Smoothing**
```
num of vertices: 2135547
num of facets: 4268725
num of boundary edges: 1167
Time of Kernel << PQGFSmoothing >> 338.622 ms. 
```
---
<img src="https://github.com/anapupa/cuMesh/blob/main/docs/images/snapshot02_L00.png" alt="Quadric Smoothed" width="480" height="whatever" /> <img src="https://github.com/anapupa/cuMesh/blob/main/docs/images/snapshot02_L01.png" alt="Original Mesh" width="480" height="whatever"/>

```cpp
#include <cuMesh/SurfaceMesh.h>
#include <cuMesh/algorithm/UpdateTopology.cuh>
#include <cuMesh/algorithm/SurfaceSmoothing.cuh>

int main (int argc, char **argv) {
    cuMesh::VFMeshData mesh_data;

    cuMesh::readPLY(argv[1], mesh_data);

    cuMesh::Topology::UpdateHalfEdge(mesh_data);
    cuMesh::Smoothing::PQGFSmoothing(mesh_data, 2, 5, 1);

    cuMesh::writePLY(argv[2], mesh_data);
}
```

## Build
```shell
git clone --recurse-submodules git@github.com:anapupa/cuMesh.git
cd cuMesh && mkdir build
cmake .. && make -j 10
```
