//
// Created by pupa on 2021/11/12.
//
#include <cuMesh/SurfaceMesh.h>
#include <cuMesh/algorithm/UpdateTopology.cuh>
#include <cuMesh/algorithm/SurfaceSmoothing.cuh>

#include <external/happly/happly.h>
#include <vector>
#include <array>
#include <helper_math.h>

int main (int argc, char **argv) {
    cuMesh::VFMeshData mesh_data;

    happly::PLYData plyIn(argv[1]);
    std::vector<std::array<double, 3>> vPos = plyIn.getVertexPositions();
    std::vector<std::vector<size_t>> fInd = plyIn.getFaceIndices<size_t>();

    thrust::host_vector<float3> mesh_v(vPos.size());
    thrust::host_vector<uint3> mesh_f(fInd.size());
    for(int i = 0; i < vPos.size(); i++)
        mesh_v[i] = make_float3(vPos[i][0], vPos[i][1], vPos[i][2]);
    for(int i = 0; i < fInd.size(); i++)
        mesh_f[i] = make_uint3(fInd[i][0], fInd[i][1], fInd[i][2]);

    mesh_data.Update(mesh_v);
    mesh_data.Update(mesh_f);

    cuMesh::Topology::UpdateHalfEdge(mesh_data, true, true);
    cuMesh::Smoothing::PQGFSmoothing(mesh_data, 2, 5, 1);

    vPos = mesh_data.GetVertexPositions();
    fInd = mesh_data.GetTriangleIndices();

//    polyscope::init();
//    polyscope::registerSurfaceMesh(argv[1],  vPos, fInd);
//    polyscope::show();

    happly::PLYData plyOut;
    plyOut.addVertexPositions(vPos);
    plyOut.addFaceIndices(fInd);
    plyOut.write(argv[2],  happly::DataFormat::Binary);

}