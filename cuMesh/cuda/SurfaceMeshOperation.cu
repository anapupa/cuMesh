//
// Created by pupa on 2021/11/13.
//
#include <cuMesh/HalfEdgeElementType.cuh>
#include "cuMesh/algorithm/SurfaceMeshOperation.cuh"
#include <probabilistic-quadrics.cuh>

namespace cuMesh::Operation{


__device__ float ComputeEdgeLength::operator()(Index hid) {
    return length(position(vertex(hid / 3, hid % 3)) - position(vertex(hid / 3, (hid + 1) % 3)));
}

__device__ float3 ComputeFaceNormal::operator()(Index fid) {
    float3 v0 = position(vertex(fid, 0));
    float3 v1 = position(vertex(fid, 1));
    float3 v2 = position(vertex(fid, 2));
    return normalize(cross(v1-v0, v2-v0));
}

__device__ Quadric ComputeFaceQuadric::operator()(Index& fid, float3& normal)  {
    float3 v0 = position(vertex(fid, 0));
    return Quadric::PlaneQuadric(v0, normal);
}

__host__ __device__ bool CheckHedgeIsBoundary::operator()(Index hid)  {
    return this->twin_halfedge(hid) == NonIndex;
}


}