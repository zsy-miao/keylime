// keylime GPU: descriptor heaps and views.
#pragma once
#include <cstdint>

namespace kl::gpu::descriptor {
// Bytes per descriptor by heap type (views, samplers, render targets, depth); 0 if unknown.
uint32_t increment(uint32_t kind);
}  // namespace kl::gpu::descriptor
