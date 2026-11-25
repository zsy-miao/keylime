// keylime: xg_xs.dll, the console's texture layout library, served by AMD's
// addrlib (texture.cpp). The title asks which swizzle mode a texture should
// use, its size, where each mip and slice lies, and copies linear data into
// tiled memory and back. Layout struct shapes and slot map as VIctory found
// them (xg.rs); the computation is AMD's own.
//
// Not yet: separate stencil planes of depth-stencil formats (plane 0 only).
#include <windows.h>

#include <algorithm>
#include <atomic>

#include "gpu/resource.h"
#include "gpu/texture.h"
#include "hle/hle.h"
#include "log.h"

namespace kl::xg {
namespace {

using gpu::resource::ResourceDesc;
namespace tx = gpu::texture;

constexpr uint32_t SWIZZLED = 0x100, ROW_MAJOR = 1;
constexpr uint32_t MIPS = 15;
constexpr uint64_t TILE = 64 << 10;

uint32_t suggested(const ResourceDesc& d) {
    return tx::default_swizzle(d.dimension == 2 ? tx::Dim::Tex1d : d.dimension == 4 ? tx::Dim::Tex3d : tx::Dim::Tex2d,
                               d.flags & 2, d.flags & 1);
}

HRESULT WINAPI XGSuggestSwizzleMode(const ResourceDesc* d, uint32_t, uint32_t* out) {
    if (!d || !out) return E_INVALIDARG;
    *out = SWIZZLED | suggested(*d);
    return S_OK;
}

struct MipOut {
    uint64_t size, offset, slice_size;
    uint32_t pitch_texels, pitch_bytes, alignment;
    uint32_t padded[3], elements[2], texels[2];
    uint32_t depth_or_array, samples, swizzle;
    uint64_t bank_rotation[2];
    uint32_t slice_depth, in_tail, tail_position;
};
struct PlaneOut {
    uint32_t usage;
    uint64_t size, slice_size, offset, alignment;
    uint32_t element_bytes, standard_mips, tail_mips;
    MipOut mips[MIPS];
};
struct LayoutOut {
    uint64_t size, alignment;
    uint32_t mips, planes;
    PlaneOut plane[4];
    uint32_t dimension;
};
static_assert(sizeof(MipOut) == 0x70 && sizeof(PlaneOut) == 0x6c8 && sizeof(LayoutOut) == 0x1b40);

struct Computer {
    const void* const* vtable;
    std::atomic<uint32_t> refs;
    ResourceDesc desc;
    tx::Layout layout;
};

// A tail mip is reported in the 256-byte mode of its kind (3 for render, which GFX10 lacks).
uint32_t tail_swizzle(uint32_t sw) {
    switch (sw) {
        case 2: case 6: case 10: case 18: case 22: case 26: return 2;
        case 23: case 27: return 3;
        case 24: return 24;
        default: return 1;
    }
}

uint32_t WINAPI add_ref(Computer* c) { return ++c->refs; }
uint32_t WINAPI release(Computer* c) {
    uint32_t left = --c->refs;
    if (!left) delete c;
    return left;
}

HRESULT WINAPI layout(Computer* c, LayoutOut* out) {
    if (!out) return E_INVALIDARG;
    memset(out, 0, sizeof *out);
    const auto& l = c->layout;
    const auto& d = c->desc;
    uint32_t mips = uint32_t(l.mips.size());
    out->size = l.size;
    out->alignment = l.alignment;
    out->mips = mips;
    out->planes = 1;
    out->dimension = d.dimension;
    PlaneOut& p = out->plane[0];
    bool depth = (d.flags & 2) && mips == 1;
    p.usage = depth ? 7 : 1;
    p.size = l.size;
    p.slice_size = l.slice_size;
    p.alignment = l.alignment;
    p.element_bytes = l.bytes_per_element;
    p.standard_mips = l.first_tail_mip;
    p.tail_mips = mips - l.first_tail_mip;
    bool three_d = d.dimension == 4;
    uint32_t array = d.depth_or_array ? d.depth_or_array : 1;
    for (uint32_t i = 0; i < mips && i < MIPS; ++i) {
        const auto& m = l.mips[i];
        MipOut& o = p.mips[i];
        o.pitch_texels = m.pitch * l.block;
        o.pitch_bytes = m.pitch * l.bytes_per_element;
        o.padded[0] = m.pitch, o.padded[1] = m.height, o.padded[2] = three_d ? m.depth : array;
        o.elements[0] = m.width_e, o.elements[1] = m.height_e;
        o.texels[0] = std::max(1u, uint32_t(d.width >> i)), o.texels[1] = std::max(1u, d.height >> i);
        o.depth_or_array = three_d ? std::max(1u, uint32_t(d.depth_or_array) >> i) : array;
        o.samples = d.sample_count ? d.sample_count : 1;
        o.slice_depth = l.block_d;
        uint64_t slice = uint64_t(m.pitch) * m.height * l.bytes_per_element;
        uint64_t dpitch = three_d ? m.depth : 1;
        if (!m.in_tail) {
            o.offset = m.offset;
            o.slice_size = slice * o.samples;
            o.size = o.slice_size * dpitch;
            o.alignment = uint32_t(l.alignment);
            o.swizzle = l.swizzle;
        } else {
            o.in_tail = 1;
            o.alignment = 256;
            o.swizzle = tail_swizzle(l.swizzle);
            if (l.thick) {
                o.slice_size = l.alignment / l.block_d;
                o.tail_position = m.tail_x | m.tail_y << 11;
            } else {
                o.offset = m.offset + m.tail_offset;
                o.slice_size = slice;
            }
            o.size = o.slice_size * dpitch;
        }
    }
    return S_OK;
}

uint64_t WINAPI element_offset(Computer* c, uint32_t plane, uint32_t mip, uint64_t x, uint32_t y, uint32_t z, uint32_t,
                               uint32_t* bit) {
    if (bit) *bit = 0;
    uint64_t at = 0;
    if (plane != 0 || !tx::element_offset(c->layout, mip, uint32_t(x), y, z, at)) return 0;
    return at;
}

HRESULT WINAPI copy_into(Computer* c, uint8_t* tiled, uint32_t plane, uint32_t sub, const uint8_t* linear, uint32_t row,
                         uint32_t depth) {
    if (!tiled || !linear || plane) return E_INVALIDARG;
    uint32_t mips = uint32_t(c->layout.mips.size());
    if (!tx::tile(c->layout, tiled, sub % mips, sub / mips, linear, row, depth)) {
        KL_WARN("xg", "copy into a texture failed (fmt %u, %llux%u, layout %x, sub %u)", c->desc.format,
                (unsigned long long)c->desc.width, c->desc.height, c->desc.layout, sub);
        return E_INVALIDARG;
    }
    return S_OK;
}
HRESULT WINAPI copy_from(Computer* c, uint8_t* linear, uint32_t plane, uint32_t sub, uint32_t, const uint8_t* tiled,
                         uint32_t row, uint32_t depth) {
    if (!tiled || !linear || plane) return E_INVALIDARG;
    uint32_t mips = uint32_t(c->layout.mips.size());
    if (!tx::detile(c->layout, tiled, sub % mips, sub / mips, linear, row, depth)) return E_INVALIDARG;
    return S_OK;
}

struct PackedMips {
    uint8_t standard, packed;
    uint32_t tiles, start;
};
struct SubresourceTiling {
    uint32_t width;
    uint16_t height, depth;
    uint32_t start;
};
// In 64 KiB tiles of plane 0, for 2D 64 KiB modes only (others fail, as on the console).
HRESULT WINAPI tiling(Computer* c, uint32_t* tiles, PackedMips* packed, uint32_t* shape, uint32_t* count, uint32_t first,
                      SubresourceTiling* out) {
    const auto& l = c->layout;
    if (l.alignment != TILE) return E_FAIL;
    if (l.thick) return E_NOTIMPL;
    uint32_t mips = uint32_t(l.mips.size()), standard = l.first_tail_mip;
    bool tail = standard < mips;
    uint64_t tail_tiles = standard == 0 ? l.slice_size / TILE : tail ? l.mips[standard - 1].offset / TILE : 0;
    if (tiles) *tiles = uint32_t(l.size / TILE);
    if (packed) *packed = {uint8_t(standard), uint8_t(mips - standard), uint32_t(tail_tiles), 0};
    if (shape) shape[0] = l.block_w * l.block, shape[1] = l.block_h * l.block, shape[2] = l.block_d;
    if (count) {
        uint32_t slices = tail ? std::min(l.slices, 2u) : l.slices;
        uint32_t total = standard * slices;
        uint32_t wanted = total > first ? std::min(total - first, *count) : 0;
        for (uint32_t i = 0; i < wanted && out; ++i) {
            uint32_t slice = (first + i) / standard, mip = (first + i) % standard;
            const auto& m = l.mips[mip];
            if (tail && slice > 0) out[i] = {};
            else out[i] = {m.pitch / l.block_w, uint16_t(m.height / l.block_h), 1,
                           uint32_t((uint64_t(slice) * l.slice_size + m.offset) / TILE)};
        }
        *count = wanted;
    }
    return S_OK;
}

HRESULT WINAPI unused(Computer* c) {
    KL_WARN("xg", "unimplemented texture computer method (fmt %u)", c->desc.format);
    return E_NOTIMPL;
}

const void* const kVtable[13] = {(void*)add_ref, (void*)release, (void*)layout, (void*)unused, (void*)unused,
                                 (void*)unused, (void*)element_offset, (void*)unused, (void*)copy_into,
                                 (void*)copy_from, (void*)tiling, (void*)unused, (void*)unused};

HRESULT WINAPI XGCreateTextureComputer(const ResourceDesc* d, Computer** out) {
    if (!d || !out) return E_INVALIDARG;
    ResourceDesc desc = *d;
    if (desc.layout == 0) desc.layout = SWIZZLED | suggested(desc);
    tx::Layout l;
    if (!gpu::resource::texture_layout(desc, l) || l.mips.size() > MIPS) {
        KL_WARN("xg", "no texture layout: dim %u %llux%ux%u fmt %u layout %x", d->dimension,
                (unsigned long long)d->width, d->height, d->depth_or_array, d->format, d->layout);
        return E_INVALIDARG;
    }
    *out = new Computer{kVtable, {1}, *d, l};
    return S_OK;
}

}  // namespace

KL_MODULE("xg", (std::vector<std::string>{"xg_xs.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(XGCreateTextureComputer), KL_EXPORT(XGSuggestSwizzleMode)}));

}  // namespace kl::xg
