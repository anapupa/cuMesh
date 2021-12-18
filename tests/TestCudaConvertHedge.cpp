//
// Created by pupa on 2021/11/12.
//
#include <cuMesh/SurfaceMesh.h>
#include <cuMesh/algorithm/UpdateTopology.h>
#include <cuMesh/algorithm/SurfaceSmoothing.h>

int main (int argc, char **argv) {
    cuMesh::VFMeshData mesh_data;

    cuMesh::readPLY(argv[1], mesh_data);

    cuMesh::Topology::UpdateHalfEdge(mesh_data);
    cuMesh::writePLY(argv[2]+std::string("0.ply"), mesh_data);
    for(int i = 0; i < 5; i++) {
        cuMesh::Smoothing::PQGFSmoothing(mesh_data, 2, 1, 1);
        cuMesh::writePLY(argv[2]+std::to_string(i+1)+".ply", mesh_data);
    }
}