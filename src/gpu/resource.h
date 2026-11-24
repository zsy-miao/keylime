// keylime GPU: heaps and resources. A resource's GPU address is its CPU
// address in title memory, as on the console.
#pragma once
#include <cstdint>

#include "gpu/object.h"
#include "gpu/texture.h"

namespace kl::gpu::resource {

constexpr uint32_t BUFFER = 1, TEX1D = 2, TEX2D = 3, TEX3D = 4;
constexpr uint32_t FLAG_RENDER_TARGET = 1, FLAG_DEPTH = 2, FLAG_UNORDERED = 4;

struct ResourceDesc {
    uint32_t dimension;
    uint64_t alignment, width;
    uint32_t height;
    uint16_t depth_or_array, mip_levels;
    uint32_t format, sample_count, sample_quality, layout, flags;
};
static_assert(sizeof(ResourceDesc) == 56);

struct HeapProperties {
    uint32_t kind, cpu_pages, pool, creation_nodes, visible_nodes;
};

struct Info {
    uint64_t address;
    ResourceDesc desc;
    uint32_t heap_kind;  // 1 default (GPU), 2 upload, 3 readback, 4 custom
};

// A title resource object's facts; false if `obj` is not one of ours.
bool info(void* obj, Info& out);
// The console texture layout for a texture description; false for buffers / unknown formats.
bool texture_layout(const ResourceDesc& d, texture::Layout& out);

}  // namespace kl::gpu::resource
