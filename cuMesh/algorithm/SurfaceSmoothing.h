//
// Created by pupa on 2021/11/13.
//
#pragma once
#include <cuMesh/HalfEdgeElementType.h>

#define Q_API __device__ __host__
#include <probabilistic-quadrics.hh>
#include <minimal-math.hh>

namespace cuMesh {

class MeshNavigators;
using Quadric = pq::quadric< pq::math<double, pq::vec<3, double>, pq::vec<3, double>, pq::mat<3, 3, double>> >;

/**
 * @brief Mesh Smoothing/Denoising
 */
namespace Smoothing {

/**
 * @brief Probabilistic Quadrics Geometry Smoothing
 * @param mesh_data
 * @param sigma_s
 * @param sigma_r
 * @param sigma_v
 */
void PQGFSmoothing(VFMeshData& mesh_data, double sigma_s, double sigma_r, double sigma_v);

}

}


namespace Kernel{
    using cuMesh::MeshNavigators;
    using cuMesh::Index;
    using cuMesh::Quadric;
    /**
     * Facet's Quadric Diffusion
     * @param navigators
     * @param tri_quadric
     * @param tri_quadric_out
     * @param sigma_s
     * @param sigma_r
     */
    __global__ void DiffuseTriQuadric(MeshNavigators navigators, Quadric* tri_quadric, Quadric* tri_quadric_out,
                                      double sigma_s, double  sigma_r);

    __global__ void UpdateVertexPosition(MeshNavigators navigators, Quadric* tri_quadric, double sigma_v);

    __global__ void UpdateVertexQuadric(MeshNavigators navi, Quadric* tri_quadric,  Quadric* vert_quadric, double sigma_v_sq);
    __global__ void UpdateVertexPositionConstraint(MeshNavigators navi, Quadric* vert_quadric, float3* f_n);
}


