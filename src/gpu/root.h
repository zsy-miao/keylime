// keylime GPU: root signatures (what the executor needs of them).
#pragma once
#include <cstdint>

namespace kl::gpu::root {
// The S# words of a root signature's static sampler at shader register `reg` (space `space`).
bool static_sampler(void* root_signature, uint32_t reg, uint32_t space, uint32_t* words);
}  // namespace kl::gpu::root
