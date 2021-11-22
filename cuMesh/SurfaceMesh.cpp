//
// Created by pupa on 2021/11/11.
//

#include "SurfaceMesh.h"
#include <cuda.h>
#include <fstream>
#include <cstring>
#include <external/happly/happly.h>
#include <vector>
#include <array>
#include <helper_math.h>
#include <cuMesh/HalfEdgeElementType.h>

namespace cuMesh {

/**
 *
 * @param file_path
 * @param vertices
 * @param triangles
 * @return
 */
bool readOBJ(const std::string file_path, thrust::host_vector<float3> &vertices, thrust::host_vector<uint3> &triangles) {
    std::ifstream fileHandle(file_path, std::ios_base::in);
    if (!fileHandle.is_open()) {
        return false;
    }
    char tmpLine[500];
    enum F_MODE {
        V, VT, VTN
    } f_mode;
    f_mode = F_MODE(VTN + 1);
    for (; fileHandle.getline(tmpLine, 500);) {
        if (tmpLine[0] == '#') continue;
        char *start;
        if ((start = strstr(tmpLine, "v "))) {
            vertices.push_back({0, 0, 0});
            float3 &xyz = *(std::prev(vertices.end()));
            sscanf(start, "v %f%f%f", &xyz.x, &xyz.y, &xyz.z);
        } else if ((start = strstr(tmpLine, "f "))) {
            triangles.push_back({0, 0, 0});
            uint3 &f = *(std::prev(triangles.end()));
            switch (f_mode) {
                case VTN:
                    sscanf(start, "f %d/%*d/%*d %d/%*d/%*d %d/%*d/%*d", &f.x, &f.y, &f.z);
                    break;
                case VT:
                    sscanf(start, "f %d/%*d %d/%*d %d/%*d", &f.x, &f.y, &f.z);
                    break;
                case V:
                    sscanf(start, "f %d %d %d", &f.x, &f.y, &f.z);
                    break;
                default:
                    if (sscanf(start, "f %d/%*d/%*d %d/%*d/%*d %d/%*d/%*d", &f.x, &f.y, &f.z) == 3)
                        f_mode = VTN;
                    else if (sscanf(start, "f %d/%*d %d/%*d %d/%*d", &f.x, &f.y, &f.z) == 3)
                        f_mode = VT;
                    else if (sscanf(start, "f %d %d %d", &f.x, &f.y, &f.z) == 3)
                        f_mode = V;
                    break;
            }
            f.x--;
            f.y--;
            f.z--;

            std::cout << f.x << ' ' << f.y << ' ' << f.z << std::endl;
        }
    }
    fileHandle.close();
    return true;
}

bool readPLY(const std::string file_path, VFMeshData& mesh_data) {
    happly::PLYData plyIn(file_path);
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
    return true;
}

bool writePLY(const std::string file_path, VFMeshData& mesh_data) {
    auto vPos = mesh_data.GetVertexPositions();
    auto fInd = mesh_data.GetTriangleIndices();

    happly::PLYData plyOut;
    plyOut.addVertexPositions(vPos);
    plyOut.addFaceIndices(fInd);
    plyOut.write(file_path,  happly::DataFormat::Binary);
    return true;
}


}