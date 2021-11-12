//
// Created by pupa on 2021/11/11.
//
#pragma once

#include "HalfEdgeElementType.cuh"

#include <thrust/sort.h>

namespace cuMesh {


namespace Topology {

void UpdateFaceFace(VFMeshData& mesh_data);

}

}


namespace Kernel{
    __global__ void ComputeVertexDegreeKernel(const uint3* face, uint32_t n_f, uint32_t * v_degree);


    __global__ void FillOutGoingHedgeKernel(const uint3* face, uint32_t n_f, uint32_t * v_offset, uint32_t* v_hedges);

    __global__ void FindHedgeTwinKernel(const uint32_t *v_offset, const uint32_t* v_hedges, const uint3 *tri,
                                        uint32_t n_h, uint32_t n_v, uint32_t* h_twin);
}