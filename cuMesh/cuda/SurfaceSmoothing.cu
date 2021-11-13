//
// Created by pupa on 2021/11/13.
//

#include <cuMesh/HalfEdgeElementType.cuh>
#include <cuMesh/HalfEdgeNavigators.cuh>
#include <cuMesh/algorithm/SurfaceSmoothing.cuh>
#include <cuMesh/algorithm/SurfaceMeshOperation.cuh>

#include <thrust/transform.h>


namespace cuMesh {

void Smoothing::PQGFSmoothing(VFMeshData &mesh_data) {
    //step1. compute triangle quadric
    MeshNavigators navigators(mesh_data);
    thrust::device_vector<Index> face_index(navigators.n_f);
    thrust::sequence(face_index.begin(), face_index.end());

    thrust::device_vector<float3>  tri_normal(navigators.n_f);
    thrust::transform(face_index.begin(), face_index.end(), tri_normal.begin(),
                      Operation::ComputeFaceNormal(mesh_data));

    thrust::device_vector<Quadric> tri_quadric(navigators.n_f);
    thrust::transform(face_index.begin(), face_index.end(), tri_normal.begin(),  tri_quadric.begin(),
                      Operation::ComputeFaceQuadric(mesh_data));
}

}