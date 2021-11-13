//
// Created by pupa on 2021/11/11.
//
#pragma once

#include <cuMesh/HalfEdgeNavigators.cuh>
#include <helper_math.h>

#define blk_size 32
inline uint32_t GetGridDim(const uint32_t& n_thread) { return  (n_thread - 1 + blk_size) / blk_size; }

namespace cuMesh {
    struct MeshNavigators: VFMeshData::PointerHub {
        __host__ MeshNavigators(VFMeshData& meshData) {
            n_v = meshData._vertices.size();
            n_f = meshData._facets.size();
            n_h = meshData._facets.size() * 3;
            _vert = meshData._vertices.data().get();
            _face = meshData._facets.data().get();
            _hedge_twin = meshData._hedges_twin.data().get();

            _v_tag = meshData._v_tags.data().get();
            _f_tag = meshData._f_tags.data().get();
        }
        // Hedge -> Face
        __host__ __device__ __forceinline__ Index static hedge2face(Index hid) {  return hid == NonIndex? hid : hid/3 ; }

        // Hedge -> Hedge
        __host__ __device__ __forceinline__ Index static next_halfedge(Index hid) { return hid / 3 * 3 + (hid + 1) % 3 ;}
        __host__ __device__ __forceinline__ Index static prev_halfedge(Index hid) { return hid / 3 * 3 + (hid + 2) % 3 ;}
        __host__ __device__ __forceinline__ Index twin_halfedge(Index hid) { return hid == NonIndex ? hid : _hedge_twin[hid];}

        // Hedge -> Vertex
        __host__ __device__ __forceinline__ Index tip_vertex(Index hid)  {
            return ((Index*)(_face + hedge2face(hid)))[(hid + 1) % 3];
        }
        __host__ __device__ __forceinline__ Index tail_vertex(Index hid)  {
            return ((Index*)(_face + hedge2face(hid)))[hid % 3];
        }


        // Face -> Face
        __host__ __device__ __forceinline__ Index adjacent_face(Index fid, size_t i_adj) {
            return hedge2face(twin_halfedge(fid * 3 + i_adj ));
        }

        // Face -> Vertex
        __host__ __device__ __forceinline__ Index vertex(Index fid, size_t i_adj) {
            return tail_vertex(fid*3+i_adj);
        }


        // Geometry Query
        __host__ __device__ __forceinline__ float3 triangle_centroid(Index fid) {
            return (_vert[_face[fid].x]+_vert[_face[fid].y]+_vert[_face[fid].z])/3.0f;
        }

        __host__ __device__ __forceinline__ float3& position(Index vid) { return _vert[vid]; }
    };
}

