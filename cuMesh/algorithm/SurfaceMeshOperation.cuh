//
// Created by pupa on 2021/11/13.
//
#pragma once
#include <cuMesh/HalfEdgeNavigators.cuh>


struct Quadric;

namespace cuMesh {


namespace Operation{



struct ComputeEdgeLength: MeshNavigators {
    explicit __host__ ComputeEdgeLength(VFMeshData& meshData): MeshNavigators(meshData){}
    __device__ float operator()(Index hid) ;
};

struct ComputeFaceNormal: MeshNavigators {
    explicit __host__ ComputeFaceNormal(VFMeshData& meshData): MeshNavigators(meshData){}
    __device__ float3 operator()(Index fid) ;
};

struct ComputeFaceQuadric: MeshNavigators {
    explicit __host__ ComputeFaceQuadric(VFMeshData& meshData): MeshNavigators(meshData){}
    __device__ Quadric operator()(Index& fid,float3& normal) ;
};





}
}