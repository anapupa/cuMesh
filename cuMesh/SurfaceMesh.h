//
// Created by pupa on 2021/11/11.
//
#pragma once
#include <cuda.h>
#include <thrust/host_vector.h>

namespace cuMesh {

struct VFMeshData;

bool readOBJ(const std::string file_path, thrust::host_vector<float3> &vertices, thrust::host_vector<uint3> &triangles);

bool readPLY(const std::string file_path, VFMeshData& meshData);
bool writePLY(const std::string file_path, VFMeshData& meshData);
}