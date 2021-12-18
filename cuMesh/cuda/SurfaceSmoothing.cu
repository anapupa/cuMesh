//
// Created by pupa on 2021/11/13.
//

#include <cuMesh/HalfEdgeElementType.h>
#include <cuMesh/HalfEdgeNavigators.cuh>
#include <cuMesh/algorithm/SurfaceSmoothing.h>
#include <cuMesh/algorithm/SurfaceMeshOperation.h>

#include <thrust/transform.h>
#include <helper_math.h>


namespace cuMesh {

void Smoothing::PQGFSmoothing(VFMeshData &mesh_data, double sigma_s, double sigma_r, double sigma_v) {
    Kernel::CudaTimeAnalysis analysis("PQGFSmoothing");
    MeshNavigators navigators(mesh_data);
    thrust::device_vector<Index> face_index(navigators.n_f);
    thrust::sequence(face_index.begin(), face_index.end());
    make_double3(1, 2, 3);
    float mean_e_len = 0;
    {
        thrust::device_vector<float> hedge_length(navigators.n_f);
        thrust::transform(face_index.begin(), face_index.end(), hedge_length.begin(),
                          Operation::ComputeEdgeLength(mesh_data));
        float total_length = thrust::reduce(hedge_length.begin(), hedge_length.end(), 0.0f);
        mean_e_len = total_length / navigators.n_f;
    }

    sigma_r *= mean_e_len;
    sigma_v *= mean_e_len;

    //step1. compute triangle quadric
    thrust::device_vector<Quadric> tri_quadric(navigators.n_f);
    thrust::transform(face_index.begin(), face_index.end(), tri_quadric.begin(),
                      Operation::ComputeFaceQuadric(mesh_data, mean_e_len*0.05));


    //step2. tri_quadric diffusion
    thrust::device_vector<Quadric> tri_quadric2(navigators.n_f);
    for(int i  = 0; i < 5; i++) {
        Kernel::DiffuseTriQuadric<<<GetGridDim(navigators.n_f), blk_size, 0>>>(
                navigators, tri_quadric.data().get(), tri_quadric2.data().get(),
                        sigma_s * sigma_s, sigma_r*sigma_r);
        thrust::swap(tri_quadric, tri_quadric2);
    }

    thrust::device_vector<Quadric> vert_quadric(navigators.n_v);
    Kernel::UpdateVertexQuadric<<<GetGridDim(navigators.n_v), blk_size, 0>>>(
        navigators, tri_quadric.data().get(),  vert_quadric.data().get(), sigma_v * sigma_v);

//    thrust::device_vector<float3> tri_nromal(navigators.n_f);
//    for(int i = 0; i < 10; i++) {
//        thrust::transform(face_index.begin(), face_index.end(), tri_nromal.begin(),
//                          Operation::ComputeFaceNormal(mesh_data));
//        Kernel::UpdateVertexPositionConstraint<<<GetGridDim(navigators.n_v), blk_size, 0>>>(
//            navigators, vert_quadric.data().get(),  tri_nromal.data().get() );
//    }

    Kernel::UpdateVertexPosition<<<GetGridDim(navigators.n_v), blk_size, 0>>>(
            navigators, tri_quadric.data().get(),  sigma_v * sigma_v);
}

}



__global__ void Kernel::DiffuseTriQuadric(MeshNavigators navi, Quadric* tri_quadric, Quadric* tri_quadric_out,
                                          double sigma_s_sq, double sigma_r_sq) {
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if(fid >= navi.n_f ) return ;
    Quadric Qf_new, Qf = tri_quadric[fid], Qff;
    double ww = 0, w, d_s, d_r, w_s, w_r, w_angle;

    double qf_norm = pq::minimal_math<double>::frobenius_inner_product(Qf.A()), qff_norm;
    float3 fid_center = navi.triangle_centroid(fid);
#pragma unroll 3
    for(int i = 0; i < 3; i++) {
        for(auto it = navi.outgoing_hedges(navi.vertex(fid, i)); !it.is_end(); ++it) {
            Index ffid = navi.hedge2face(*it);
            if( ffid == NonIndex) continue;
            Qff = tri_quadric[ffid] ;
            qff_norm = pq::minimal_math<double>::frobenius_inner_product(Qff.A());

            float3 _v0 = navi.position( navi.vertex(ffid, 0) );
            float3 _v1 = navi.position( navi.vertex(ffid, 1) );
            float3 _v2 = navi.position( navi.vertex(ffid, 2) );

            d_s = length( fid_center - (_v0+_v1+_v2)/3.0f );
            d_s = ::exp(-0.5 * d_s *d_s / sigma_s_sq )* length(navi.cross_dot(ffid));

            d_r = (Qf(_v0.x, _v0.y, _v0.z)+Qf(_v1.x, _v1.y, _v1.z)+Qf(_v2.x, _v2.y, _v2.z))
                  * length( navi.cross_dot(ffid) ) / 6.0f;
            d_r = exp(-0.5*d_r/sigma_r_sq);
            w_angle = pq::minimal_math<double>::trace_of_product(Qff.A(), Qf.A())/sqrt(qf_norm*qff_norm);

////            printf("[face %d]: |A|: %lf, |B|: %lf > <A,B>/|A||B|: %lf\n", fid, qf_norm, qff_norm, d_r);
//
            w_angle = (w_angle > 0.11)?w_angle:0; //
//            d_r =  w_angle = 1;
            w = max(d_r * d_s * w_angle, 0.0001);
            Qf_new += Qff * w;
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
            Qff = tri_quadric[ffid] ;
            qff_norm = pq::minimal_math<double>::frobenius_inner_product(Qff.A());

            float3 _v0 = navi.position( navi.vertex(ffid, 0) );
            float3 _v1 = navi.position( navi.vertex(ffid, 1) );
            float3 _v2 = navi.position( navi.vertex(ffid, 2) );

            d_s = length( fid_center - (_v0+_v1+_v2)/3.0f );
            d_s = ::exp(-0.5 * d_s *d_s / sigma_s_sq ) * length(navi.cross_dot(ffid));

            w = ::exp(-0.5 * d_s *d_s / sigma_s_sq ) * ::exp(-0.5 * d_r / sigma_r_sq );
            Qf_new += tri_quadric[ffid] * w;
            ww += w;
        }
    }

    assert(abs(ww) > 0.0001 );
    tri_quadric_out[fid] =  Qf_new / ww;
}


__global__ void Kernel::UpdateVertexPosition(MeshNavigators navi, Quadric* tri_quadric, double sigma_v_sq) {
    uint32_t vid =  blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= navi.n_v || navi.is_v_boundary(vid) ) return ;
    float3 p = navi.position(vid);
    Quadric Qv;
    double w, ww = 0;
    for(auto it = navi.outgoing_hedges(vid); !it.is_end(); ++it) {
        Index ffid = navi.hedge2face(*it);
        if( ffid == NonIndex ) continue;
        float3 _v0 = navi.position( navi.vertex(ffid, 0) );
        float3 _v1 = navi.position( navi.vertex(ffid, 1) );
        float3 _v2 = navi.position( navi.vertex(ffid, 2) );

        Quadric q = Quadric::point_quadric( cuMesh::to_pq((_v0+_v1+_v2)/3.0f  ) );
        w = length( navi.cross_dot(ffid) );

        Qv += (tri_quadric[ffid]) * w;
        ww += w;

    }
    Qv += Quadric::point_quadric(cuMesh::to_pq(p)) * 0.005;

    auto min_p = Qv.minimizer();
//    printf("%f %f %f\n", min_p.x, min_p.y, min_p.z);

    if(min_p.x == min_p.x && min_p.y == min_p.y && min_p.z == min_p.z) {
//        printf("%f %f %f --> %f %f %f\n", p.x, p.y, p.z, min_p.x, min_p.y, min_p.z);
        navi._vert[vid] = make_float3(min_p.x, min_p.y, min_p.z);
    }
    else  printf("failed\n");
}



__global__ void Kernel::UpdateVertexQuadric(MeshNavigators navi, Quadric* tri_quadric,  Quadric* vert_quadric, double sigma_v_sq) {
    uint32_t vid =  blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= navi.n_v || navi.is_v_boundary(vid) ) return ;
    float3 p = navi.position(vid);
    Quadric Qv;
    double w, ww = 0;
    for(auto it = navi.outgoing_hedges(vid); !it.is_end(); ++it) {
        Index ffid = navi.hedge2face(*it);
        if( ffid == NonIndex ) continue;
//        float3 _v0 = navi.position( navi.vertex(ffid, 0) );
//        float3 _v1 = navi.position( navi.vertex(ffid, 1) );
//        float3 _v2 = navi.position( navi.vertex(ffid, 2) );
//
//        Quadric q = Quadric::point_quadric( cuMesh::to_pq((_v0+_v1+_v2)/3.0f  ) );
        w = length( navi.cross_dot(ffid) );

        Qv += (tri_quadric[ffid]) * w;
        ww += w;

    }
    vert_quadric[vid] = Qv/ww + Quadric::point_quadric(cuMesh::to_pq(p)) * 0.005;
}

__global__ void  Kernel::UpdateVertexPositionConstraint(MeshNavigators navi, Quadric* vert_quadric, float3* f_n){
    uint32_t vid =  blockIdx.x * blockDim.x + threadIdx.x;
    if(vid >= navi.n_v || navi.is_v_boundary(vid) ) return ;

    double ww = 0;
    float3 v_motion{0 ,0 ,0};
    for(auto it = navi.outgoing_hedges(vid); !it.is_end(); ++it) {
        Index ffid = navi.hedge2face(*it);
        if( ffid == NonIndex ) continue;
        double tri_area = length(navi.cross_dot(ffid));
        double lambda = vert_quadric[vid].constraint_minimizer(cuMesh::to_pq(navi.position(vid)), cuMesh::to_pq(f_n[ffid]));
        v_motion += lambda * f_n[ffid] * tri_area;
        ww += tri_area;
    }
    navi._vert[vid] += v_motion/ww;
}
