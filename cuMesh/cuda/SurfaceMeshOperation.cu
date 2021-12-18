//
// Created by pupa on 2021/11/13.
//
#include <cuMesh/HalfEdgeElementType.h>

#include "cuMesh/algorithm/SurfaceMeshOperation.h"
#include "cuMesh/algorithm/SurfaceSmoothing.h"

float3 Q_API cuMesh::from_pq(const pq::dvec3 & v) {
    return make_float3(v.x, v.y, v.z);
}

pq::dvec3 Q_API cuMesh::to_pq(const float3& v) {
    return pq::dvec3 {v.x, v.y, v.z};
}

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

__device__ float3 ComputeVertexNormal::operator()(Index vid) {
    float3 normal_sum{0 ,0,0};
    for(auto it = outgoing_hedges(vid); !it.is_end(); ++it) {
        Index ffid = hedge2face(*it);
        normal_sum += this->cross_dot(ffid);
    }
}

__device__ Quadric ComputeFaceQuadric::operator()(Index& fid)  {
    float3 v0 = position(vertex(fid, 0));
    float3 v1 = position(vertex(fid, 1));
    float3 v2 = position(vertex(fid, 2));

//    return cuMesh::Quadric::plane_quadric(to_pq(v0), to_pq(normal));
    return Quadric::probabilistic_plane_quadric(to_pq((v0+v1+v2)/3.0f), to_pq(normal(fid)), _stddev, 0.1);
//    return Quadric::probabilistic_triangle_quadric(to_pq(v0), to_pq(v1), to_pq(v2), _stddev);
}

__host__ __device__ bool CheckHedgeIsBoundary::operator()(Index hid)  {
    return this->twin_halfedge(hid) == NonIndex;
}


}