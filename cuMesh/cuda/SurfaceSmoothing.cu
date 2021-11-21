//
// Created by pupa on 2021/11/13.
//

#include <cuMesh/HalfEdgeElementType.cuh>
#include <cuMesh/HalfEdgeNavigators.cuh>
#include <cuMesh/algorithm/SurfaceSmoothing.cuh>
#include <cuMesh/algorithm/SurfaceMeshOperation.cuh>

#include <thrust/transform.h>
#include <helper_math.h>


namespace cuMesh {

void Smoothing::PQGFSmoothing(VFMeshData &mesh_data, double sigma_s, double sigma_r, double sigma_v) {
    Kernel::CudaTimeAnalysis analysis("PQGFSmoothing");
    MeshNavigators navigators(mesh_data);
    thrust::device_vector<Index> face_index(navigators.n_f);
    thrust::sequence(face_index.begin(), face_index.end());

    float mean_e_len = 0;
    {
        thrust::device_vector<float> hedge_length(navigators.n_f);
        thrust::transform(face_index.begin(), face_index.end(), hedge_length.begin(),
                          Operation::ComputeEdgeLength(mesh_data));
        float total_length = thrust::reduce(hedge_length.begin(), hedge_length.end(), 0.0f);
        mean_e_len = total_length / navigators.n_f;
    }

    sigma_r *= mean_e_len;
    sigma_s *= mean_e_len;
    sigma_v *= mean_e_len;


    //step1. compute triangle quadric
    thrust::device_vector<float3>  tri_normal(navigators.n_f);
    thrust::transform(face_index.begin(), face_index.end(), tri_normal.begin(),
                      Operation::ComputeFaceNormal(mesh_data));
    thrust::device_vector<Quadric> tri_quadric(navigators.n_f);
    thrust::transform(face_index.begin(), face_index.end(), tri_normal.begin(),  tri_quadric.begin(),
                      Operation::ComputeFaceQuadric(mesh_data, mean_e_len*0.05));


    //step2. tri_quadric diffusion
    thrust::device_vector<Quadric> tri_quadric2(navigators.n_f);
    for(int i  = 0; i < 3; i++) {
        Kernel::DiffuseTriQuadric<<<GetGridDim(navigators.n_f), blk_size, 0>>>(
                navigators, tri_quadric.data().get(), tri_quadric2.data().get(),
                        sigma_s * sigma_s, sigma_r*sigma_r);
        thrust::swap(tri_quadric, tri_quadric2);
    }

    Kernel::UpdateVertexPosition<<<GetGridDim(navigators.n_v), blk_size, 0>>>(
            navigators, tri_quadric.data().get(),  sigma_v * sigma_v);
}

}



__global__ void Kernel::DiffuseTriQuadric(MeshNavigators navi, Quadric* tri_quadric, Quadric* tri_quadric_out,
                                          float sigma_s_sq, float sigma_r_sq) {
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if(fid > navi.n_f ) return ;
    Quadric Qf_new, Qf = tri_quadric[fid];
    float ww = 0, w, d_s, d_r;
    float3 fid_center = navi.triangle_centroid(fid);
#pragma unroll 3
    for(int i = 0; i < 3; i++) {
        for(auto it = navi.outgoing_hedges(navi.vertex(fid, i)); !it.is_end(); ++it) {
            Index ffid = navi.hedge2face(*it);
            if( ffid == NonIndex) continue;
            float3 _v0 = navi.position( navi.vertex(ffid, 0) );
            float3 _v1 = navi.position( navi.vertex(ffid, 1) );
            float3 _v2 = navi.position( navi.vertex(ffid, 2) );

            d_s = length( fid_center - (_v0+_v1+_v2)/3.0f );
            d_r = (Qf(_v0)+Qf(_v1)+Qf(_v2)) * length( navi.cross_dot(ffid) ) / 6.0f;

            w = ::exp(-0.5 * d_s *d_s / sigma_s_sq ) * ::exp(-0.5 * d_r / sigma_r_sq );
            Qf_new += tri_quadric[ffid] * w;
            ww += w;
        }
    }

#pragma unroll 3
    for(int i = 0; i < 3; i++) {
        Index hid = navi.twin_halfedge(fid * 3 + i);
        if(hid == NonIndex) continue;
        Index vid = navi.tip_vertex(navi.next_halfedge(hid));
        for(auto it = navi.outgoing_hedges(vid); !it.is_end(); ++it) {
            Index ffid = navi.hedge2face(*it);
            if( ffid == NonIndex) continue;
            float3 _v0 = navi.position( navi.vertex(ffid, 0) );
            float3 _v1 = navi.position( navi.vertex(ffid, 1) );
            float3 _v2 = navi.position( navi.vertex(ffid, 2) );

            d_s = length( fid_center - (_v0+_v1+_v2)/3.0f );
            d_r = (Qf(_v0)+Qf(_v1)+Qf(_v2)) * length( navi.cross_dot(ffid) ) / 6.0f;

            w = ::exp(-0.5 * d_s *d_s / sigma_s_sq ) * ::exp(-0.5 * d_r / sigma_r_sq );
            Qf_new += tri_quadric[ffid] * w;
            ww += w;
        }
    }

    tri_quadric_out[fid] = Qf_new / ww;
}


__global__ void Kernel::UpdateVertexPosition(MeshNavigators navi, Quadric* tri_quadric, double sigma_v_sq) {
    uint32_t vid =  blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= navi.n_v || navi.is_v_boundary(vid) ) return ;
    float3 p = navi.position(vid);
    Quadric Qv;
    float w, ww = 0;
    for(auto it = navi.outgoing_hedges(vid); !it.is_end(); ++it) {
        Index ffid = navi.hedge2face(*it);
        if( ffid == NonIndex ) continue;
        float3 _v0 = navi.position( navi.vertex(ffid, 0) );
        float3 _v1 = navi.position( navi.vertex(ffid, 1) );
        float3 _v2 = navi.position( navi.vertex(ffid, 2) );

        float3 diff = p - (_v0+_v1+_v2)/3.0f ;
        w = ::exp(-0.5 * dot(diff, diff) / sigma_v_sq ) ;

        Qv += tri_quadric[ffid] * w;
        ww += w;
        Qv += Quadric::PointQuadric(p) * w * 0.05f;
    }
    Qv /= ww;

    float3 min_p = Qv.minimizer();
//    printf("%f %f %f\n", min_p.x, min_p.y, min_p.z);

    if(min_p.x == min_p.x && min_p.y == min_p.y && min_p.z == min_p.z) {
//        printf("%f %f %f --> %f %f %f\n", p.x, p.y, p.z, min_p.x, min_p.y, min_p.z);
        navi._vert[vid] = min_p;
    }
//    else  printf("failed\n");
}
