#include <cuMesh/HalfEdgeElementType.cuh>

#include <cuMesh/algorithm/UpdateTopology.cuh>
#include <cuMesh/HalfEdgeNavigators.cuh>
#include <thrust/scan.h>

using cuMesh::MeshNavigators;


#define DEBUG_CHECK

namespace cuMesh {

struct isDeletedTag{
    __host__ __device__ bool operator()(const VFMeshData::Tag &tag) { return tag == VFMeshData::Tag::DELETED;}
};

void Topology::UpdateFaceFace(VFMeshData &mesh_data, bool is_manifold) {
    uint32_t n_f = mesh_data._facets.size();

    MeshNavigators navigators(mesh_data);
    // 1.1 count each vertex's degree
    thrust::device_vector<uint32_t> v_degree(mesh_data._vertices.size(), 0);

    Kernel::CountVertexDegree<<<(n_f - 1 + blk_size) / blk_size, blk_size, 0>>>(
            mesh_data._facets.data().get(), n_f, v_degree.data().get() );
    // 1.2 arrange gpu memory and each vertex's offset
    thrust::device_vector<uint32_t> v_offset(mesh_data._vertices.size(), 0);
    thrust::exclusive_scan(v_degree.begin(), v_degree.end(), v_offset.begin());
    thrust::device_vector<uint32_t> unordered_hedge(mesh_data._facets.size() * 3, NonIndex);

    Kernel::FillOutGoingHedge<<<GetGridDim(n_f * 3), blk_size, 0>>>(
            mesh_data._facets.data().get(), n_f, v_offset.data().get(), unordered_hedge.data().get() );

    thrust::exclusive_scan(v_degree.begin(), v_degree.end(), v_offset.begin());

    Kernel::FindHedgeTwin<<<GetGridDim(n_f * 3), blk_size, 0>>>(
            navigators, v_offset.data().get(), unordered_hedge.data().get() );

    if(!is_manifold) {
        Kernel::TagNonManifoldTriangle<<<GetGridDim(n_f * 3), blk_size, 0>>>(navigators);
        thrust::stable_sort_by_key(mesh_data._f_tags.begin(), mesh_data._f_tags.end(), mesh_data._facets.begin());

        uint32_t n_deleted = thrust::count_if(mesh_data._f_tags.begin(), mesh_data._f_tags.end(), isDeletedTag());
        std::cout << "deleted faces: " << n_deleted <<std::endl;
        mesh_data.Update(mesh_data._facets.begin(), mesh_data._facets.end() - n_deleted);

        return UpdateFaceFace(mesh_data, true);
    }

    thrust::device_vector<uint32_t> n_boundary_hedge(1, 0);
    Kernel::CountBoundaryEdges<<<GetGridDim(n_f * 3), blk_size, 0>>>(
        navigators, n_boundary_hedge.data().get() );

    std::cout << "num of boundary edges: " << n_boundary_hedge[0] << std::endl;
}

}



__global__ void Kernel::CountVertexDegree(const uint3* face, uint32_t n_f, uint32_t * v_degree) {
    uint32_t fid =   blockIdx.x * blockDim.x + threadIdx.x;
    if(  fid >= n_f )  return ;
    atomicAdd(v_degree + face[fid].x, 1);
    atomicAdd(v_degree + face[fid].y, 1);
    atomicAdd(v_degree + face[fid].z, 1);
}

__global__ void Kernel::FillOutGoingHedge(const uint3* face, uint32_t n_f, uint32_t * v_offset, uint32_t* v_hedges) {
    uint32_t fid =   blockIdx.x * blockDim.x + threadIdx.x;
    if(  fid >= n_f )  return ;
    v_hedges[atomicAdd(v_offset + face[fid].x, 1)] = fid *3;
    v_hedges[atomicAdd(v_offset + face[fid].y, 1)] = fid *3 + 1;
    v_hedges[atomicAdd(v_offset + face[fid].z, 1)] = fid *3 + 2;
}


__global__ void Kernel::FindHedgeTwin(MeshNavigators navigators, const uint32_t *v_offset, const uint32_t* v_hedges) {
    uint32_t hid =   blockIdx.x * blockDim.x + threadIdx.x;
    if(  hid >= navigators.n_h )  return ;
    uint32_t vTail = navigators.tail_vertex(hid);
    uint32_t vTip  = navigators.tip_vertex(hid);

    uint32_t vTip_HBegin = v_offset[vTail];
    uint32_t vTip_HEnd   = (vTail+1 == navigators.n_v ) ? navigators.n_h: v_offset[vTail+1];

    uint32_t n_of_twins = 0, n_of_repeat = 0;
    for(uint32_t i = vTip_HBegin; i != vTip_HEnd; i++) {
        uint32_t new_tip = navigators.tip_vertex(v_hedges[i]);
        n_of_repeat += (navigators.tip_vertex(v_hedges[i]) == vTip);
        if(vTail == new_tip) {
            navigators._hedge_twin[hid] = v_hedges[i];
            n_of_twins ++ ;
        }
    }
    if(n_of_repeat > 1 || n_of_twins >= 2)  navigators._hedge_twin[hid] = hid;

#ifdef DEBUG_CHECK
    assert(n_of_twins <= 1);
    if(n_of_twins == 1) {
        uint32_t hid_twin = navigators.twin_halfedge(hid);
        uint32_t twin_tail = navigators.tail_vertex(hid_twin);
        uint32_t twin_tip = navigators.tip_vertex(hid_twin);
        if(!(twin_tail == vTip &&  twin_tip == vTail)) {
            printf("%d %d // %d %d \n", vTip, vTail, twin_tip, twin_tail);
        }
        assert(twin_tail == vTip &&  twin_tip == vTail);
    }
#endif
}

__global__ void Kernel::TagNonManifoldTriangle(MeshNavigators navigators) {
    uint32_t hid =   blockIdx.x * blockDim.x + threadIdx.x;
    if(  hid >= navigators.n_h )  return ;
    if(navigators.twin_halfedge(hid) == hid) {
//        printf("%d \n", navigators.hedge2face(hid));
        navigators._f_tag[navigators.hedge2face(hid)] = cuMesh::VFMeshData::Tag::DELETED;
    }
}

__global__ void Kernel::CountBoundaryEdges(MeshNavigators navigators, uint32_t* n_boundary_hedges) {
    uint32_t hid =  blockIdx.x * blockDim.x + threadIdx.x;
    if(  hid >= navigators.n_h )  return ;
    if(navigators.twin_halfedge(hid) == NonIndex) {
        atomicAdd(n_boundary_hedges, 1);
//        printf("l %d %d\n", navigators.tip_vertex(hid)+1, navigators.tail_vertex(hid)+1);
    }
}

