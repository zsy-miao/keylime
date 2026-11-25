// keylime shaders: a compute shader that copies indirect dispatch arguments to a tight array,
// zeroing those at or past the GPU-side count (so every one can be issued as an indirect dispatch).
#pragma once
#include <cstdint>
#include <vector>

namespace kl::shader {
// Push constants: u64 source, u64 destination, u64 count address, u32 source stride, u32 max.
std::vector<uint32_t> dispatch_clamp();
}  // namespace kl::shader
