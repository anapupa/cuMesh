#pragma once
#include <cuda.h>
#include <thrust/host_vector.h>

namespace cuMesh {

struct VFMeshData;


/**
 * @brief Example class to demonstrate the features of the custom CSS.
 * @param file_path : path
 * @param vertices : vertices
 * @param triangles : triangles
 * @return
 */
bool readOBJ(const std::string file_path, thrust::host_vector<float3> &vertices, thrust::host_vector<uint3> &triangles);

/**
 * @brief Example class to demonstrate the features of the custom CSS.
 * @param file_path
 * @param meshData
 * @return
 */
bool readPLY(const std::string file_path, VFMeshData& meshData);

/**
 * @brief Example class to demonstrate the features of the custom CSS.
 * @param file_path
 * @param meshData
 * @return
 */
bool writePLY(const std::string file_path, VFMeshData& meshData);
}