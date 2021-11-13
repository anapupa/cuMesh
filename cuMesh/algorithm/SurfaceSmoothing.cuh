//
// Created by pupa on 2021/11/13.
//
#pragma once
#include <cuMesh/HalfEdgeElementType.cuh>
#include <probabilistic-quadrics.cuh>

namespace cuMesh {

class MeshNavigators;

namespace Smoothing {

void PQGFSmoothing(VFMeshData& mesh_data);

}

}


