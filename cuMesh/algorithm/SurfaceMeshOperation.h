//
// Created by pupa on 2021/11/13.
//
#pragma once
#include <cuMesh/HalfEdgeNavigators.cuh>

#define Q_API __device__ __host__
#include <probabilistic-quadrics.hh>
#include <minimal-math.hh>
// Helper struct for thrust

namespace cuMesh {

using Quadric = pq::quadric< pq::math<double, pq::vec<3, double>, pq::vec<3, double>, pq::mat<3, 3, double>> >;
float3 Q_API from_pq(const pq::dvec3& v) ;

pq::dvec3 Q_API to_pq(const float3& v) ;
/**
 * @brief function struct for thrust
 */
namespace Operation{

struct ComputeEdgeLength: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    __device__ float operator()(Index hid) ;
};

struct ComputeFaceNormal: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    __device__ float3 operator()(Index fid) ;
};

struct ComputeVertexNormal: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    __device__ float3 operator()(Index vid) ;
};

struct ComputeFaceQuadric: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    float _stddev{0};
    __host__ ComputeFaceQuadric(VFMeshData& meshData, float stddev): MeshNavigators(meshData), _stddev(stddev) {}
    __device__ Quadric operator()(Index& fid) ;
};

struct CheckTriangleIsDeleted: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    __device__ bool operator()(Index& fid) {
        return (_f_tag[fid] & VFMeshData::DELETED) == VFMeshData::DELETED;
    }
};

struct ComputeMeshEdgeLength: MeshNavigators {
    using MeshNavigators::MeshNavigators;
    __device__ float operator()(Index& fid) {
        return (length(_vert[vertex(fid, 0)]-_vert[vertex(fid, 1)]) +
                length(_vert[vertex(fid, 1)]-_vert[vertex(fid, 2)]) +
                length(_vert[vertex(fid, 2)]-_vert[vertex(fid, 0)]) ) /3.0f;
    }
};

template<uint32_t TagValue>
struct CheckTagHasValue{
    __host__ __device__ bool operator()(const uint32_t &tag) { return (tag & TagValue) == TagValue;}
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

/**
 * @brief cuda kernel namespace
 */
namespace Kernel{
    struct CudaTimeAnalysis{
        CudaTimeAnalysis(std::string&& name): __name(name) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaEventRecord(_start,0);
        };

        ~CudaTimeAnalysis() {
            cudaEventRecord(_stop, 0);
            cudaEventSynchronize(_stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, _start, _stop);
//            std::swap(_start, _stop);
            std::cout << "Time of Kernel << " << __name << " >> " << milliseconds << " ms. " << std::endl;
        };

    private:
        cudaEvent_t _start, _stop;
        std::string __name;
    };
}