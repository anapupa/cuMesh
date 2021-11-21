//
// Created by pupa on 2021/11/13.
//
#pragma once
#include <cuMesh/HalfEdgeElementType.cuh>
#include <probabilistic-quadrics.cuh>

namespace cuMesh {

class MeshNavigators;

namespace Smoothing {

void PQGFSmoothing(VFMeshData& mesh_data, double sigma_s, double sigma_r, double sigma_v);

}

}


namespace Kernel{
    using cuMesh::MeshNavigators;
    using cuMesh::Index;
    __global__ void DiffuseTriQuadric(MeshNavigators navigators, Quadric* tri_quadric, Quadric* tri_quadric_out,
                                      float sigma_s, float sigma_r);

    __global__ void UpdateVertexPosition(MeshNavigators navigators, Quadric* tri_quadric, double sigma_v);

}


