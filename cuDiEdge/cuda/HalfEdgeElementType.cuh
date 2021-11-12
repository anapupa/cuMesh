//
// Created by pupa on 2021/11/11.
//
#pragma once

#include <thrust/device_vector.h>

namespace cuMesh {

using Index = uint32_t;
#define NonIndex 0xffffffff

    struct VFMeshData{
        enum class ElementFlag{
            DELETED    = 0x0001,		// This bit indicate that the vertex is deleted from the mesh
            NOTREAD    = 0x0002,		// This bit indicate that the vertex of the mesh is not readable
            NOTWRITE   = 0x0004,		// This bit indicate that the vertex is not modifiable
            MODIFIED   = 0x0008,		// This bit indicate that the vertex is modified
            VISITED    = 0x0010,		// This bit can be used to mark the visited vertex
            SELECTED   = 0x0020,		// This bit can be used to select
            BORDER     = 0x0100,        // Border Flag
            USER0      = 0x0200			// First user bit
        };
        thrust::device_vector<float3>  _vertices;
        thrust::device_vector<uint3>   _facets;
        thrust::device_vector<Index>   _hedges_twin;

        struct PointerHub {
            float3          *_vert{nullptr};
            uint3           *_face{nullptr};
            Index           *_hedge_twin{nullptr};
            uint32_t        n_v{0}, n_f{0};
        };
    };
}

