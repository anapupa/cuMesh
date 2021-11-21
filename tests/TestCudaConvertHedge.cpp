//
// Created by pupa on 2021/11/12.
//
#include <cuMesh/SurfaceMesh.h>
#include <cuMesh/algorithm/UpdateTopology.cuh>
#include <cuMesh/algorithm/SurfaceSmoothing.cuh>

int main (int argc, char **argv) {
    cuMesh::VFMeshData mesh_data;

    cuMesh::readPLY(argv[1], mesh_data);

    cuMesh::Topology::UpdateHalfEdge(mesh_data);
    cuMesh::Smoothing::PQGFSmoothing(mesh_data, 2, 5, 1);

    cuMesh::writePLY(argv[2], mesh_data);
}