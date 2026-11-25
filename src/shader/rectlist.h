// keylime shaders: a geometry shader that turns the console's rectangle lists into strips.
#pragma once
#include <cstdint>
#include <vector>

namespace kl::shader {
// SPIR-V for a geometry shader passing gl_Position and vec4 varyings at `locations`.
std::vector<uint32_t> rect_geometry(const std::vector<uint8_t>& locations, bool quad = false);
}  // namespace kl::shader
