// keylime GPU: small device children (query heaps, command signatures).
#pragma once
#include <array>
#include <cstdint>
#include <vector>

namespace kl::gpu::misc {
// A command signature's byte stride and arguments ({type, three fields} each); false if `obj` is not one.
bool signature(void* obj, uint32_t& stride, std::vector<std::array<uint32_t, 4>>& arguments);
}  // namespace kl::gpu::misc
