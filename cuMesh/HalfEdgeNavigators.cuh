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
            _v_out_h = meshData._v_out_h.data().get();

            _v_tag = meshData._v_tags.data().get();
            _f_tag = meshData._f_tags.data().get();
        }

        class HedgeAroundVertexCirculator{
        public:
            //! default constructor
            __device__ __host__ HedgeAroundVertexCirculator(const MeshNavigators& navigators, Index vid)
                    : mesh_(navigators), is_end_(false), curr_hid(navigators._v_out_h[vid]) {
                begin_hid = curr_hid;
            }

            HedgeAroundVertexCirculator& __device__ __host__ operator++(){
                curr_hid = mesh_.ccw_rotate(curr_hid);
                if(curr_hid == begin_hid) is_end_ = true;
                return *this;
            }

            //! get the vertex the circulator refers to
            Index __device__ __host__ operator*() const {
                return curr_hid;
            }

            // helper for C++11 range-based for-loops
            __device__ __host__ bool is_end(){
                return curr_hid == NonIndex || is_end_;
            }

        private:
            const MeshNavigators& mesh_;
            Index curr_hid, begin_hid;
            bool is_end_; // helper for C++11 range-based for-loops
        };

        // Hedge -> Face
        __host__ __device__ __forceinline__ Index static hedge2face(Index hid) {  return hid == NonIndex? hid : hid/3 ; }

        // Hedge -> Hedge
        __host__ __device__ __forceinline__ Index static next_halfedge(Index hid) { return hid / 3 * 3 + (hid + 1) % 3 ;}
        __host__ __device__ __forceinline__ Index static prev_halfedge(Index hid) { return hid / 3 * 3 + (hid + 2) % 3 ;}
        __host__ __device__ __forceinline__ Index twin_halfedge(Index hid) const{ return hid == NonIndex ? hid : _hedge_twin[hid];}
        __host__ __device__ __forceinline__ Index ccw_rotate(Index hid) const{
            return twin_halfedge(prev_halfedge(hid));
        }
        __host__ __device__ __forceinline__ Index cw_rotate(Index hid) const{
            Index twin_hid = twin_halfedge(hid);
            return twin_hid == NonIndex ? NonIndex: next_halfedge(hid);
        }


        // Hedge -> Vertex
        __host__ __device__ __forceinline__ Index& tip_vertex(Index hid)  {
            return ((Index*)(_face + hedge2face(hid)))[(hid + 1) % 3];
        }
        __host__ __device__ __forceinline__ Index& tail_vertex(Index hid)  {
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
        __host__ __device__ __forceinline__ float3 triangle_centroid(Index fid) const {
            return (_vert[_face[fid].x]+_vert[_face[fid].y]+_vert[_face[fid].z])/3.0f;
        }

        __host__ __device__ __forceinline__ float3 normal(Index fid) {
            float3 v0 = position(vertex(fid, 0));
            float3 v1 = position(vertex(fid, 1));
            float3 v2 = position(vertex(fid, 2));
            return normalize(cross(v1-v0, v2-v0));
        }

        __host__ __device__ __forceinline__ float3 cross_dot(Index fid) {
            float3 v0 = position(vertex(fid, 0));
            float3 v1 = position(vertex(fid, 1));
            float3 v2 = position(vertex(fid, 2));
            return cross(v1-v0, v2-v0);
        }

        __host__ __device__ __forceinline__ bool is_v_boundary(Index vid) const{
            return (_v_tag[vid] & VFMeshData::DELETED) == VFMeshData::DELETED;
        }

        __host__ __device__ __forceinline__ float3& position(Index vid) { return _vert[vid]; }

        __host__ __device__ HedgeAroundVertexCirculator outgoing_hedges(Index vid) const {
            return HedgeAroundVertexCirculator(*this, vid);
        }

    };



}

