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
    uint64_t offset;            // byte offset of the mip in one slice (tail mips: of the tail block)
    uint32_t pitch, height, depth;  // padded, in elements
    uint32_t width_e, height_e;     // real size in elements
    bool in_tail;
    uint32_t tail_offset;       // byte offset inside the tail block
    uint32_t tail_x, tail_y, tail_z;  // element position inside the tail block
};

struct Layout {
    uint64_t size = 0, alignment = 0, slice_size = 0;
    uint32_t bytes_per_element = 0, block = 1;  // block = 4 for BC formats
    std::vector<Mip> mips;
    uint32_t first_tail_mip = 0;      // == mips.size() when there is no tail
    uint32_t block_w = 0, block_h = 0, block_d = 1;  // swizzle block in elements
    uint32_t slices = 1, swizzle = 0;
    bool thick = false;
    Desc desc;
    // Depth-stencil formats: a separate stencil plane (8-bit elements) after the depth plane.
    bool has_stencil = false;
    uint64_t stencil_offset = 0, stencil_size = 0;
};

// Bytes per element and element width in texels (4 for block compression); false if unknown.
bool element(uint32_t format, uint32_t& bytes, uint32_t& block);
uint32_t full_mip_count(uint32_t w, uint32_t h, uint32_t d);
// The console layout of `d`; false for an invalid or unsupported one.
bool layout(const Desc& d, Layout& out);

// Linear <-> tiled copies of one subresource (mip, slice), as the console's XG does.
bool tile(const Layout& l, void* surface, uint32_t mip, uint32_t slice, const void* linear, uint64_t row, uint64_t depth_pitch);
bool detile(const Layout& l, const void* surface, uint32_t mip, uint32_t slice, void* linear, uint64_t row, uint64_t depth_pitch);
// Byte offset of an element from the surface start; false if outside.
bool element_offset(const Layout& l, uint32_t mip, uint32_t x, uint32_t y, uint32_t z, uint64_t& out);

// The swizzle mode the console picks for a D3D layout value of 0, by usage.
uint32_t default_swizzle(Dim dim, bool depth, bool render_target);

}  // namespace kl::gpu::texture
