// keylime shaders: a compute shader that copies ExecuteIndirect arguments to a tight array,
// zeroing commands at or past the GPU-side count and those whose sizes no frame would use
// (stale or unwritten argument memory), so a bad argument cannot hang the GPU.
#pragma once
#include <cstdint>
#include <vector>

namespace kl::shader {
// type: 0 draw (4 dwords), 1 indexed draw (5), 2 dispatch (3), 3 mesh dispatch (3).
// Push constants: u64 source, u64 destination, u64 count address, u32 source stride, u32 max.
// max_x: the most groups in x a dispatch may ask for (the host GPU's limit).
std::vector<uint32_t> indirect_clamp(uint32_t type, uint32_t max_x = 65535);
}  // namespace kl::shader
