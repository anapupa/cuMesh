# cuPMP
cuPMP is a simple C++/Cuda(thrust) parallel geometry processing library.

We develop it as a toolbox for post-processing of Delaunay 3d-reconstruction.

#Build
```shell
git clone --recurse-submodules git@github.com:anapupa/cuMesh.git
cd cuMesh && mkdir build
cmake .. && make -j 10
```