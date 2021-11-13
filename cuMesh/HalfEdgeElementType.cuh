//
// Created by pupa on 2021/11/11.
//
#pragma once

#include <thrust/device_vector.h>

namespace cuMesh {

using Index = uint32_t;
#define NonIndex 0xffffffff

    struct VFMeshData{
        enum class Tag{
            NONE       = 0x0000,
            USER0      = 0x0200,		// First user bit
            NOTREAD    = 0x0002,		// This bit indicate that the vertex of the mesh is not readable
            NOTWRITE   = 0x0004,		// This bit indicate that the vertex is not modifiable
            MODIFIED   = 0x0008,		// This bit indicate that the vertex is modified
            VISITED    = 0x0010,		// This bit can be used to mark the visited vertex
            SELECTED   = 0x0020,		// This bit can be used to select
            BORDER     = 0x0100,        // Border Flag
            DELETED    = 0x8000,		// This first bit indicate that the vertex is deleted from the mesh
        };

        thrust::device_vector<float3>  _vertices;
        thrust::device_vector<uint3>   _facets;
        thrust::device_vector<Tag>     _v_tags;
        thrust::device_vector<Tag>     _f_tags;
        thrust::device_vector<Index>   _hedges_twin;

        void Update(const thrust::host_vector<float3>& V) {
            _vertices = V;
            _v_tags   = thrust::host_vector<Tag>(_vertices.size(), Tag::NONE);
        }

        void Update(const thrust::host_vector<uint3>& F) {
            _facets = F ;
            _f_tags   = thrust::host_vector<Tag>(_facets.size(), Tag::NONE);
            _hedges_twin = thrust::host_vector<Index>(_facets.size() * 3, NonIndex);
        }

        void Update(decltype(_vertices)::iterator begin, decltype(_vertices)::iterator end) {
            _vertices = decltype(_vertices) (begin, end);
            _v_tags   = thrust::host_vector<Tag>(_vertices.size(), Tag::NONE);
        }

        void Update(decltype(_facets)::iterator begin, decltype(_facets)::iterator end) {
            _facets = decltype(_facets) (begin, end);
            _f_tags   = thrust::host_vector<Tag>(_facets.size(), Tag::NONE);
            _hedges_twin = thrust::host_vector<Index>(_facets.size() * 3, NonIndex);
        }

        std::vector<std::array<double, 3>> GetVertexPositions() {
            std::vector<std::array<double, 3>> positions(_vertices.size());
            thrust::host_vector<float3> vertices = _vertices;
            for(int i = 0; i < vertices.size(); i++)
                positions[i] = {vertices[i].x, vertices[i].y, vertices[i].z};
            return positions;
        }

        std::vector<std::vector<size_t>> GetTriangleIndices() {
            std::vector<std::vector<size_t>> triangles(_facets.size(), std::vector<size_t>(3, 0));
            thrust::host_vector<uint3> facets = _facets;
            for(int i = 0; i < facets.size(); i++){
                triangles[i][0] = facets[i].x;
                triangles[i][1] = facets[i].y;
                triangles[i][2] = facets[i].z;
            }
            return triangles;
        }


        struct PointerHub {
            float3          *_vert{nullptr};
            uint3           *_face{nullptr};
            Tag             *_v_tag{nullptr};
            Tag             *_f_tag{nullptr};
            Index           *_hedge_twin{nullptr};
            uint32_t        n_v{0}, n_f{0}, n_h{0};
        };
    };
}

