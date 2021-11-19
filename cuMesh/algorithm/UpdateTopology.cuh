//
// Created by pupa on 2021/11/11.
//
#pragma once

#include <cuMesh/HalfEdgeElementType.cuh>

#include <thrust/sort.h>

namespace cuMesh {

class MeshNavigators;

namespace Topology {

void UpdateHalfEdge(VFMeshData& mesh_data, bool is_e_manifold=false, bool is_v_manifold=false);

void RemoveNonManifoldTriangle(VFMeshData& mesh_data);

void RepairNonManifoldVertex(VFMeshData& mesh_data);

}

}


namespace Kernel {
    using cuMesh::MeshNavigators;

    struct CudaTimeAnalysis{
        CudaTimeAnalysis(std::string&& name): __name(name) {
            cudaEventCreate(&_start);
            cudaEventCreate(&_stop);
            cudaEventRecord(_start);
        };

        ~CudaTimeAnalysis() {
            cudaEventRecord(_stop);
            cudaEventSynchronize(_stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, _start, _stop);
            std::swap(_start, _stop);
            std::cout << "Time of Kernel << " << __name << " >> " << milliseconds << " ms. " << std::endl;
        };

      private:
        cudaEvent_t _start, _stop;
        std::string __name;
    };

    __global__ void CountVertexDegree(const uint3* face, uint32_t n_f, uint32_t * v_degree);

    __global__ void FillOutGoingHedge(const uint3* face, uint32_t n_f, uint32_t * v_offset, uint32_t* v_hedges);

    __global__ void FindHedgeTwin(MeshNavigators navigators, const uint32_t *v_offset, const uint32_t* v_hedges);

    __global__ void TagNonManifoldTriangle(MeshNavigators navigators);

    __global__ void TagMeshBorder(MeshNavigators navigators);

    __global__ void SplitNonManifoldBVertex(MeshNavigators navigators, uint32_t* bnd_h_new_tail, uint32_t* global_vid);
}