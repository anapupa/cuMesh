//
// Created by pupa on 2021/11/13.
//
#pragma once
#include <cuMesh/HalfEdgeNavigators.cuh>

// Helper struct for thrust

struct Quadric;

namespace cuMesh {


namespace Operation{



struct ComputeEdgeLength: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    __device__ float operator()(Index hid) ;
};

struct ComputeFaceNormal: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    __device__ float3 operator()(Index fid) ;
};

struct ComputeFaceQuadric: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    __device__ Quadric operator()(Index& fid,float3& normal) ;
};


template<uint32_t TagValue>
struct CheckTagHasValue{
    __host__ __device__ bool operator()(const uint32_t &tag) { return tag == TagValue;}
};

template<class ValueType, ValueType value>
struct IsEqual{
    __host__ __device__ bool operator()(const ValueType &x) { return x == value;}
};




// Check whether hedge is boundary according to it's twin
struct CheckHedgeIsBoundary: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    __host__ __device__ bool operator()(Index hid) ;
};


}
}