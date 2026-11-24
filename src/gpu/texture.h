// keylime GPU: console texture layouts (RDNA 2 / GFX10.3) from AMD's own
// address library (addrlib, MIT, vendored from Mesa): size, alignment and
// where every mip / slice lives, exactly as the console's layout library
// (XG, built on addrlib) computes them.
#pragma once
#include <cstdint>
#include <vector>

namespace kl::gpu::texture {

enum class Dim : uint8_t { Tex1d, Tex2d, Tex3d };

struct Desc {
    Dim dim = Dim::Tex2d;
    uint32_t width = 1, height = 1, depth_or_array = 1, mips = 1, samples = 1;
    uint32_t format = 0;   // DXGI format number (console extras included)
    uint32_t swizzle = 0;  // addrlib ADDR_SW_* number
};

struct Mip {
    uint64_t offset;            // byte offset of the mip in one slice
    uint32_t pitch, height, depth;  // in elements
    bool in_tail;
};

struct Layout {
    uint64_t size = 0, alignment = 0, slice_size = 0;
    uint32_t bytes_per_element = 0, block = 1;  // block = 4 for BC formats
    std::vector<Mip> mips;
};

// Bytes per element and element width in texels (4 for block compression); false if unknown.
bool element(uint32_t format, uint32_t& bytes, uint32_t& block);
uint32_t full_mip_count(uint32_t w, uint32_t h, uint32_t d);
// The console layout of `d`; false for an invalid or unsupported one.
bool layout(const Desc& d, Layout& out);

// The swizzle mode the console picks for a D3D layout value of 0, by usage.
uint32_t default_swizzle(Dim dim, bool depth, bool render_target);

}  // namespace kl::gpu::texture
