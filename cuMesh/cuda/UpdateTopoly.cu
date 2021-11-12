#include <cuMesh/cuda/HalfEdgeElementType.cuh>

#include "UpdateTopology.cuh"
#include <thrust/scan.h>

constexpr static int blk_size = 8;

__global__ void Kernel::ComputeVertexDegreeKernel(const uint3* face, uint32_t n_f, uint32_t * v_degree) {
    uint32_t fid =   blockIdx.x * blockDim.x + threadIdx.x;
    if(  fid >= n_f )  return ;
    atomicAdd(v_degree + face[fid].x, 1);
    atomicAdd(v_degree + face[fid].y, 1);
    atomicAdd(v_degree + face[fid].z, 1);
}

__global__ void Kernel::FillOutGoingHedgeKernel(const uint3* face, uint32_t n_f, uint32_t * v_offset,
                                        uint32_t* v_hedges) {
    uint32_t fid =   blockIdx.x * blockDim.x + threadIdx.x;
    if(  fid >= n_f )  return ;
    v_hedges[atomicAdd(v_offset + face[fid].x, 1)] = fid *3;
    v_hedges[atomicAdd(v_offset + face[fid].y, 1)] = fid *3 + 1;
    v_hedges[atomicAdd(v_offset + face[fid].z, 1)] = fid *3 + 2;
}


__global__ void Kernel::FindHedgeTwinKernel(const uint32_t *v_offset, const uint32_t* v_hedges, const uint3 *tri,
                                    uint32_t n_h, uint32_t n_v, uint32_t* h_twin) {
    uint32_t hid =   blockIdx.x * blockDim.x + threadIdx.x;
    if(  hid >= n_h )  return ;
    uint32_t vTail = ((uint32_t*)(tri + hid /3))[hid % 3];
    uint32_t vTip  = ((uint32_t*)(tri + hid /3))[(hid+1) % 3];

    printf(" %d -> %d \n", vTip, vTail);
    uint32_t vTip_HBegin = v_offset[vTip];
    uint32_t vTip_HEnd   = vTip == (n_v - 1) ? n_h: v_offset[vTip+1];
    uint32_t n_of_twins = 0;
    for(uint32_t i = vTip_HBegin; i != vTip_HEnd; i++) {
        uint32_t new_hid = v_hedges[i];
        uint32_t new_tip = ((uint32_t*)(tri + new_hid /3))[(new_hid+1) % 3];
        if(vTail == new_tip) {
            h_twin[hid] = new_hid;
            n_of_twins ++ ;
        }
    }
    assert(n_of_twins <= 1 && "non-manifold edges");
}

namespace cuMesh {

void Topology::UpdateFaceFace(VFMeshData &mesh_data) {
    mesh_data._hedges_twin.resize(mesh_data._facets.size());
    uint32_t n_f = mesh_data._facets.size();
    // 1.1 count each vertex's degree
    thrust::device_vector<uint32_t> v_degree(mesh_data._vertices.size(), 0);

    Kernel::ComputeVertexDegreeKernel<<<(n_f - 1 + blk_size) / blk_size, blk_size, 0>>>(
            mesh_data._facets.data().get(), n_f, v_degree.data().get() );
    // 1.2 arrange gpu memory and each vertex's offset
    thrust::device_vector<uint32_t> v_offset(mesh_data._vertices.size(), 0);
    thrust::exclusive_scan(v_degree.begin(), v_degree.end(), v_offset.begin());
    thrust::device_vector<uint32_t> unordered_hedge(mesh_data._facets.size() * 3, NonIndex);


    Kernel::FillOutGoingHedgeKernel<<<(n_f - 1 + blk_size) / blk_size, blk_size, 0>>>(
            mesh_data._facets.data().get(), n_f, v_offset.data().get(), unordered_hedge.data().get() );

    thrust::exclusive_scan(v_degree.begin(), v_degree.end(), v_offset.begin());

    Kernel::FindHedgeTwinKernel<<<(n_f * 3 - 1 + blk_size) / blk_size, blk_size, 0>>>(
            v_offset.data().get(), unordered_hedge.data().get(), mesh_data._facets.data().get(),
            n_f*3, mesh_data._vertices.size(), mesh_data._hedges_twin.data().get() );
}

}

