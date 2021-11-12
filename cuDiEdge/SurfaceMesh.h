//
// Created by pupa on 2021/11/11.
//
#pragma once
#include <cuda.h>
#include <thrust/host_vector.h>


bool readOBJ(std::string file_path, thrust::host_vector<float3>& vertices, thrust::host_vector<int3>& triangles);
