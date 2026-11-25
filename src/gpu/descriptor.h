// keylime GPU: descriptor heaps and views.
#pragma once
#include <cstdint>

namespace kl::gpu::descriptor {
// Bytes per descriptor by heap type (views, samplers, render targets, depth); 0 if unknown.
uint32_t increment(uint32_t kind);
// A descriptor heap object's type (0 views, 1 samplers, 2 RTV, 3 DSV) and start address.
bool heap_info(void* heap, uint32_t& kind, uint64_t& start);
// A render target / depth view record.
struct Target {
    void* resource;
    uint32_t format, mip, layer;
    bool depth;
};
bool target_info(uint64_t handle, Target& out);
// S# words for a root signature's static sampler (13-dword D3D12_STATIC_SAMPLER_DESC).
void sampler_from_static(const uint32_t* desc13, uint32_t* words);
}  // namespace kl::gpu::descriptor
