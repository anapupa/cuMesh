# cuMesh
cuMesh is a simple C++/Cuda(thrust) parallel geometry processing library.

We develop it as a toolbox for post-processing of Delaunay 3d-reconstruction.

## Mesh Smoothing
```
num of vertices: 2135547
num of facets: 4268725
num of boundary edges: 1167
Time of Kernel << PQGFSmoothing >> 338.622 ms. 
```
---
<img src="https://github.com/anapupa/cuMesh/blob/main/docs/images/snapshot-2_L00.png" alt="Quadric Smoothed" width="500" height="whatever" /> 
<img src="https://github.com/anapupa/cuMesh/blob/main/docs/images/snapshot-2_L01.png" alt="Original Mesh" width="500" height="whatever"/>

## Build
```shell
git clone --recurse-submodules git@github.com:anapupa/cuMesh.git
cd cuMesh && mkdir build
cmake .. && make -j 10
```