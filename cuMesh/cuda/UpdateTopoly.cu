#include <cuMesh/HalfEdgeElementType.cuh>

#include <cuMesh/algorithm/UpdateTopology.cuh>
#include <cuMesh/algorithm/SurfaceMeshOperation.cuh>

#include <cuMesh/HalfEdgeNavigators.cuh>
#include <thrust/scan.h>


using cuMesh::MeshNavigators;
using cuMesh::Operation::CheckHedgeIsBoundary;
using cuMesh::VFMeshData;
using cuMesh::Operation::CheckTagHasValue;


#define DEBUG_CHECK

namespace cuMesh {

void Topology::UpdateHalfEdge(VFMeshData &mesh_data, bool is_e_manifold, bool is_v_manifold) {
    {
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
    }
    if(!is_e_manifold) {
        RemoveNonManifoldTriangle(mesh_data);
        return UpdateHalfEdge(mesh_data, true, is_v_manifold);
    }else if(!is_v_manifold) {
        RepairNonManifoldVertex(mesh_data);
        return UpdateHalfEdge(mesh_data, true, true);
    }else {
        MeshNavigators navigators(mesh_data);
        Kernel::TagMeshBorder<<<GetGridDim(navigators.n_h), blk_size, 0>>>(navigators);
        Operation::IsEqual<uint32_t, uint32_t(NonIndex)> is_bnd_hedge;
        uint32_t n_bnd_h = thrust::count_if(mesh_data._hedges_twin.begin(), mesh_data._hedges_twin.end(), is_bnd_hedge);
        std::cout << "num of boundary edges: " << n_bnd_h << std::endl;
    }
}

void Topology::RemoveNonManifoldTriangle(VFMeshData& mesh_data) {
    MeshNavigators navigators(mesh_data);
    Kernel::TagNonManifoldTriangle<<<GetGridDim(navigators.n_f * 3), blk_size, 0>>>(navigators);
    thrust::stable_sort_by_key(mesh_data._f_tags.begin(), mesh_data._f_tags.end(), mesh_data._facets.begin());

    uint32_t n_deleted = thrust::count_if(mesh_data._f_tags.begin(), mesh_data._f_tags.end(), CheckTagHasValue<VFMeshData::DELETED>());
    std::cout << "num of non-manifold faces: " << n_deleted <<std::endl;

    mesh_data._facets.erase(mesh_data._facets.end() - n_deleted, mesh_data._facets.end());
    mesh_data._f_tags.erase(mesh_data._f_tags.end() - n_deleted, mesh_data._f_tags.end());
    mesh_data._hedges_twin.erase(mesh_data._hedges_twin.end() - n_deleted*3, mesh_data._hedges_twin.end());
}

void Topology::RepairNonManifoldVertex(VFMeshData& mesh_data) {
    assert(mesh_data._facets.size()*3 == mesh_data._hedges_twin.size());
    //step1. count boundary vertices and boundary hedges.
    MeshNavigators navigators(mesh_data);
    CheckHedgeIsBoundary hedge_bnd_check(mesh_data);
    Operation::CheckTagHasValue<VFMeshData::BORDER> has_border_tag;
    Kernel::TagMeshBorder<<<GetGridDim(navigators.n_h), blk_size, 0>>>(navigators);
    uint32_t n_bnd_h = thrust::count_if(mesh_data._hedges_twin.begin(), mesh_data._hedges_twin.end(), hedge_bnd_check);
    uint32_t n_bnd_v = thrust::count_if(mesh_data._v_tags.begin(), mesh_data._v_tags.end(), has_border_tag);

    uint32_t n_old_v = mesh_data._vertices.size();
    mesh_data._vertices.insert(mesh_data._vertices.end(), n_bnd_h - n_bnd_v, make_float3(0, 0, 0));
    mesh_data._v_tags.insert(mesh_data._v_tags.end(), n_bnd_h - n_bnd_v, VFMeshData::BORDER);
    mesh_data._v_out_h.insert(mesh_data._v_out_h.end(), n_bnd_h - n_bnd_v, NonIndex);

    navigators = MeshNavigators(mesh_data);
    thrust::device_vector<Index> bnd_h_new_tail(navigators.n_h, NonIndex);
    thrust::device_vector<Index> global_vid(1, n_old_v);
    Kernel::SplitNonManifoldBVertex<<<GetGridDim(navigators.n_h), blk_size, 0>>>(
            navigators, bnd_h_new_tail.data().get(), global_vid.data().get() );
}

}

__global__ void Kernel::SplitNonManifoldBVertex(MeshNavigators navigators, uint32_t *bnd_h_new_tail, uint32_t *global_vid) {
    uint32_t hid = blockIdx.x * blockDim.x + threadIdx.x;
    if (hid >= navigators.n_h || navigators.twin_halfedge(hid) != NonIndex) return;

    uint32_t v_tail = navigators.tail_vertex(hid);
    //check the boundary hedge's original tail whether is locked. if not, map original tail to the global new one
    if (atomicCAS(navigators._v_tag + v_tail, VFMeshData::BORDER, VFMeshData::NOTREAD) != VFMeshData::BORDER) {
        uint32_t new_vid = atomicAdd(global_vid, 1);
        navigators.position( new_vid ) = navigators.position(v_tail);
        //ccw_rotate hid and update it's face table
        for( cuMesh::Index curr_hid = hid; curr_hid != NonIndex; curr_hid = navigators.ccw_rotate(curr_hid)) {
            navigators.tail_vertex(curr_hid) = new_vid;
        }
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

    uint32_t vTip_HBegin = v_offset[vTip];
    uint32_t vTip_HEnd   = (vTip+1 == navigators.n_v ) ? navigators.n_h: v_offset[vTip+1];

    uint32_t n_of_twins = 0, n_of_repeat = 0;
    for(uint32_t i = vTip_HBegin; i != vTip_HEnd; i++) {
        uint32_t new_tip = navigators.tip_vertex(v_hedges[i]);
        n_of_repeat += (navigators.tip_vertex(v_hedges[i]) == vTip);
        if(vTail == new_tip) {
            navigators._hedge_twin[hid] = v_hedges[i];
            n_of_twins ++ ;
        }
    }

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
    if(n_of_repeat > 1 || n_of_twins >= 2)  navigators._hedge_twin[hid] = hid;
}

__global__ void Kernel::TagNonManifoldTriangle(MeshNavigators navigators) {
    uint32_t hid =   blockIdx.x * blockDim.x + threadIdx.x;
    if(  hid >= navigators.n_h )  return ;
    if(navigators.twin_halfedge(hid) == hid) {
        navigators._f_tag[navigators.hedge2face(hid)] = cuMesh::VFMeshData::DELETED;
    }
}

__global__ void Kernel::TagMeshBorder(MeshNavigators navigators) {
    uint32_t hid = blockIdx.x * blockDim.x + threadIdx.x;
    if(  hid >= navigators.n_h )  return ;
    if(navigators.twin_halfedge(hid) == NonIndex ) {
        navigators._f_tag[navigators.hedge2face(hid)] = cuMesh::VFMeshData::BORDER;
        navigators._v_tag[navigators.tail_vertex(hid)] = cuMesh::VFMeshData::BORDER;
    }
}
