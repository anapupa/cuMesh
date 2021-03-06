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

    cuMesh::writePLY(argv[2], mesh_data);
}