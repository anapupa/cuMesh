//
// Created by pupa on 2021/11/11.
//

#include "SurfaceMesh.h"
#include <cuda.h>
#include <fstream>
#include <cstring>


bool readOBJ(std::string file_path, thrust::host_vector<float3>& vertices, thrust::host_vector<uint3>& triangles){
    std::ifstream fileHandle(file_path,std::ios_base::in);
    if(!fileHandle.is_open() ) {
        return false;
    }
    char tmpLine[500];
    enum F_MODE{V, VT, VTN} f_mode;
    f_mode = F_MODE(VTN + 1);
    for(;fileHandle.getline(tmpLine,500);){
        if ( tmpLine[0] == '#' ) continue;
        char *start;
        if((start=strstr(tmpLine,"v "))){
            vertices.push_back({0, 0, 0});
            float3& xyz = *(std::prev(vertices.end()));
            sscanf(start,"v %f%f%f",&xyz.x,&xyz.y,&xyz.z);
        }else if((start=strstr(tmpLine,"f "))){
            triangles.push_back({0, 0, 0});
            uint3& f = *(std::prev(triangles.end()));
            switch (f_mode) {
                case VTN:
                    sscanf(start,"f %d/%*d/%*d %d/%*d/%*d %d/%*d/%*d",&f.x,&f.y,&f.z);
                    break;
                case VT:
                    sscanf(start,"f %d/%*d %d/%*d %d/%*d",&f.x,&f.y,&f.z);
                    break;
                case V:
                    sscanf(start,"f %d %d %d",&f.x,&f.y,&f.z);
                    break;
                default:
                    if(sscanf(start,"f %d/%*d/%*d %d/%*d/%*d %d/%*d/%*d",&f.x,&f.y,&f.z)==3)
                        f_mode = VTN;
                    else if(sscanf(start,"f %d/%*d %d/%*d %d/%*d",&f.x,&f.y,&f.z)==3)
                        f_mode = VT;
                    else if(sscanf(start,"f %d %d %d",&f.x,&f.y,&f.z)==3)
                        f_mode = V;
                    break;
            }
            f.x --; f.y --; f.z--;

            std::cout << f.x << ' ' <<f.y << ' ' <<f.z <<std::endl;
        }
    }
    fileHandle.close();
    return true;
}

