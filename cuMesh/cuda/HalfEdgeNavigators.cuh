//
// Created by pupa on 2021/11/11.
//
#pragma once

#include "HalfEdgeElementType.cuh"

namespace cuMesh {
    struct MeshNavigators: VFMeshData::PointerHub {
        __host__ MeshNavigators(VFMeshData& meshData) {
            n_v = meshData._vertices.size();
            n_f = meshData._facets.size();
            _vert = meshData._vertices.data().get();
            _face = meshData._facets.data().get();
            _hedge_twin = meshData._hedges_twin.data().get();
        }

        // Hedge -> Face
        __host__ __device__ Index hedge2face(Index hid) {  return hid == NonIndex? hid : hid/3 ; }

        // Hedge -> Hedge
        __host__ __device__ Index next_halfedge(Index hid) { return hid / 3 * 3 + (hid + 1) % 3 ;}
        __host__ __device__ Index prev_halfedge(Index hid) { return hid / 3 * 3 + (hid + 2) % 3 ;}
        __host__ __device__ Index twin_halfedge(Index hid) { return hid == NonIndex ? hid : _hedge_twin[hid];}

        // Hedge -> Vertex
        __host__ __device__ Index tail_vertex(Index hid)  {
            return ((Index*)(_face + hedge2face(hid)))[(hid + 1) % 3];
        }
        __host__ __device__ Index tip_vertex(Index hid)  {
            return ((Index*)(_face + hedge2face(hid)))[hid % 3];
        }

        // Face -> Face
        __host__ __device__ Index adjacent_face(Index fid, size_t i_adj) {
            return hedge2face(twin_halfedge(fid * 3 + i_adj ));
        }
    };
}

