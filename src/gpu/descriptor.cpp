// keylime GPU: descriptor heaps. Handles are plain addresses (CPU and GPU
// alike, as on the console). Sizes: views 32 bytes and samplers 16 (RDNA 2's
// image and sampler descriptor sizes, so a title copying descriptors as bytes
// moves whole ones); render target / depth views 128 (ours alone).
//
// keylime's plan (differs from VIctory): descriptor memory will hold real
// RDNA 2 descriptor words, which translated shaders read natively through
// raw addresses -- no mirroring of title heaps into host heaps every frame.
#include <windows.h>

#include <algorithm>
#include <cstring>

#include "gpu/device.h"
#include "gpu/descriptor.h"
#include "gpu/fence.h"
#include "gpu/images.h"
#include "gpu/memory.h"
#include "gpu/resource.h"
#include "gpu/texture.h"
#include "log.h"

namespace kl::gpu::descriptor {
namespace {

const Iid HEAP_IID = guid(0x8efb471d, 0x616c, 0x4f49, {0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51});

struct HeapDesc {
    uint32_t kind, count, flags, nodes;
};
// Newer titles pass the description followed by memory of their own: address + size
// (or a null address to have the heap allocate).
struct Placement {
    uint64_t address, size;
};

struct Heap {
    Object object;
    HeapDesc desc;
    uint64_t start;
    bool owned;
};

HRESULT WINAPI query(Heap* h, const Iid* iid, void** out) {
    bool ok = same(iid, HEAP_IID) || same(iid, fence::PAGEABLE_IIDS[0]) || same(iid, fence::PAGEABLE_IIDS[1]) ||
              same(iid, fence::PAGEABLE_IIDS[2]);
    return answer(Kind::DescriptorHeap, iid, ok ? h : nullptr, out);
}
void WINAPI get_desc(Heap* h, HeapDesc* out) { if (out) *out = h->desc; }
uint64_t WINAPI get_start(Heap* h) { return h->start; }

void** heap_table() {
    static void** t = [] {
        void** t = make_table(Kind::DescriptorHeap, 24);
        fill_object(t, (void*)&query);
        fill_child(t);
        t[8] = (void*)&get_desc;   // 0x40
        t[9] = (void*)&get_start;  // 0x48 CPU
        t[10] = (void*)&get_start; // 0x50 GPU
        return t;
    }();
    return t;
}

HRESULT WINAPI create_heap(void*, const HeapDesc* d, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    uint32_t inc = increment(d->kind);
    if (!inc) return E_INVALIDARG;
    uint64_t size = uint64_t(inc) * (d->count ? d->count : 1);
    // Read as a hint only: a title passing the plain description has other data after it.
    Placement p;
    memcpy(&p, d + 1, sizeof p);
    bool placed = p.address && p.size == size;
    auto* h = new Heap{{heap_table(), 0, {0}}, *d, 0, !placed};
    if (placed) {
        h->start = p.address;
    } else {
        // Below 4 GiB: shaders form table addresses from 32 bits (the high dword is zero).
        MEM_ADDRESS_REQUIREMENTS req{};
        req.HighestEndingAddress = reinterpret_cast<void*>(uintptr_t(0xffffffffull));
        MEM_EXTENDED_PARAMETER param{};
        param.Type = MemExtendedParameterAddressRequirements;
        param.Pointer = &req;
        void* m = VirtualAlloc2(GetCurrentProcess(), nullptr, size_t((size + 0xffff) & ~0xffffull), MEM_COMMIT | MEM_RESERVE,
                                PAGE_READWRITE, &param, 1);
        if (!m) {
            KL_WARN("gpu", "no room below 4 GiB for a descriptor heap of %llu bytes", (unsigned long long)size);
            m = VirtualAlloc(nullptr, size_t(size), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        }
        if (!m) return E_OUTOFMEMORY;
        h->start = uint64_t(m);
    }
    if (h->start >> 32) KL_WARN("gpu", "descriptor heap at %llx is above 4 GiB: shaders cannot reach it", (unsigned long long)h->start);
    memory::ensure(h->start, size);
    KL_TRACE("gpu", "descriptor heap: type %u, %u descriptors, flags %u, %s at %llx", d->kind, d->count, d->flags,
             placed ? "title memory" : "ours", (unsigned long long)h->start);
    return query(h, iid, out);
}

uint32_t WINAPI increment_size(void*, uint32_t kind) { return increment(kind); }

// ------------------------------------------------------------- views ----
// Real RDNA 2 words (docs/rdna2-descriptors.md). Image FORMAT holds the DXGI
// format for now (keylime-internal until the GFX10 IMG_FORMAT table lands).

uint32_t dst_sel(uint32_t mapping, int c) {
    // D3D12 4-component mapping: 3 bits per channel (0-3 = source channel, 4 = 0, 5 = 1);
    // 0x1688 is the default identity.
    uint32_t v = (mapping >> (3 * c)) & 7;
    if (v <= 3) return 4 + v;
    return v == 5 ? 1 : 0;
}

void buffer_words(uint32_t* w, uint64_t address, uint32_t stride, uint32_t records, uint32_t format) {
    w[0] = uint32_t(address);
    w[1] = uint32_t(address >> 32) & 0xffff;
    w[1] |= (stride & 0x3fff) << 16;
    w[2] = records;
    w[3] = 4 | 5 << 3 | 6 << 6 | 7 << 9;  // xyzw
    w[3] |= (format & 0x7f) << 12;
    w[3] |= 1u << 24;  // RESOURCE_LEVEL (1 on GFX10)
    w[3] |= 3u << 28;  // OOB_SELECT raw
}

// T# types.
enum : uint32_t { T1D = 8, T2D = 9, T3D = 10, TCUBE = 11, T1DA = 12, T2DA = 13, T2DMS = 14, T2DMSA = 15 };

void image_words(uint32_t* w, const resource::Info& r, uint32_t format, uint32_t type, uint32_t base_level,
                 uint32_t levels, uint32_t base_array, uint32_t arrays, uint32_t mapping) {
    memset(w, 0, 32);
    texture::Layout l;
    resource::texture_layout(r.desc, l);
    uint32_t width = uint32_t(r.desc.width), height = r.desc.height;
    uint32_t mips = r.desc.mip_levels ? r.desc.mip_levels : uint32_t(l.mips.size());
    uint64_t a = r.address >> 8;
    w[0] = uint32_t(a);
    w[1] = uint32_t(a >> 32) & 0xff;
    w[1] |= (format & 0x1ff) << 20;
    w[1] |= ((width - 1) & 3) << 30;
    w[2] = ((width - 1) >> 2) & 0xfff;
    w[2] |= ((height - 1) & 0x3fff) << 14;
    w[2] |= 1u << 31;
    for (int c = 0; c < 4; ++c) w[3] |= dst_sel(mapping, c) << (3 * c);
    if (levels == ~0u || base_level + levels > mips) levels = mips - base_level;
    w[3] |= (base_level & 0xf) << 12;
    w[3] |= ((base_level + levels - 1) & 0xf) << 16;
    uint32_t sw = r.desc.layout == 1 ? 0 : (r.desc.layout & ~0xffu) == 0x100 ? (r.desc.layout & 0xff)
                : texture::default_swizzle(r.desc.dimension == resource::TEX1D ? texture::Dim::Tex1d
                                            : r.desc.dimension == resource::TEX3D ? texture::Dim::Tex3d : texture::Dim::Tex2d,
                                            r.desc.flags & resource::FLAG_DEPTH, r.desc.flags & resource::FLAG_RENDER_TARGET);
    w[3] |= (sw & 0x1f) << 20;
    w[3] |= (type & 0xf) << 28;
    uint32_t depth = r.desc.dimension == resource::TEX3D ? r.desc.depth_or_array : (arrays ? arrays : r.desc.depth_or_array);
    w[4] = (depth - 1) & 0x1fff;
    w[4] |= (base_array & 0x1fff) << 16;
    w[5] = ((mips - 1) & 0xf) << 4;  // MAX_MIP
}

// A buffer view's V#: structured (stride), raw (byte-addressed, stride 0) or typed (element size).
void buffer_view(uint32_t* w, uint64_t address, uint32_t format, uint64_t first, uint32_t count, uint32_t stride, bool raw) {
    if (stride) {
        buffer_words(w, address + first * stride, stride, count, 0);
    } else if (raw || !format) {
        buffer_words(w, address + first * 4, 0, count * 4, 0);
    } else {
        uint32_t bytes = 4, block = 1;
        texture::element(format, bytes, block);
        buffer_words(w, address + first * bytes, bytes, count, format);
    }
    memory::ensure(address + first * (stride ? stride : 4), 1);
}

struct CbvDesc { uint64_t location; uint32_t size; uint32_t pad; };
void WINAPI create_cbv(void*, const CbvDesc* d, uint64_t handle) {
    if (!handle || !d) return;
    auto* w = reinterpret_cast<uint32_t*>(handle);
    memset(w, 0, 32);
    // Constant buffers are 16-byte rows (shaders index them as structured buffers of float4).
    buffer_words(w, d->location, 16, (d->size + 15) / 16, 0);
    memory::ensure(d->location, d->size ? d->size : 1);
}

struct SrvDesc {
    uint32_t format, dim, mapping, pad;
    union {
        struct { uint64_t first; uint32_t count, stride, flags; } buffer;
        struct { uint32_t most_detailed, levels, first_array, array_size, plane; float min_lod; } tex;
    };
};
static_assert(sizeof(SrvDesc) == 40);

void WINAPI create_srv(void*, void* res, const SrvDesc* d, uint64_t handle) {
    if (!handle) return;
    auto* w = reinterpret_cast<uint32_t*>(handle);
    memset(w, 0, 32);
    resource::Info r;
    if (!res) return;  // null view: all zero reads zero
    if (!resource::info(res, r)) return;
    if (r.desc.dimension == resource::BUFFER) {
        if (d) buffer_view(w, r.address, d->format, d->buffer.first, d->buffer.count, d->buffer.stride, d->buffer.flags & 1);
        else buffer_view(w, r.address, 0, 0, uint32_t(r.desc.width / 4), 0, true);
        return;
    }
    uint32_t format = d && d->format ? d->format : r.desc.format;
    images::viewed(res, format);
    uint32_t mapping = d ? d->mapping : 0x1688;
    uint32_t type = T2D, bl = 0, lv = ~0u, ba = 0, as = 0;
    if (d) {
        switch (d->dim) {
            case 2: type = T1D; bl = d->tex.most_detailed; lv = d->tex.levels; break;
            case 3: type = T1DA; bl = d->tex.most_detailed; lv = d->tex.levels; ba = d->tex.first_array; as = d->tex.array_size; break;
            case 4: type = T2D; bl = d->tex.most_detailed; lv = d->tex.levels; break;
            case 5: type = T2DA; bl = d->tex.most_detailed; lv = d->tex.levels; ba = d->tex.first_array; as = d->tex.array_size; break;
            case 6: type = T2DMS; break;
            case 7: type = T2DMSA; ba = d->tex.most_detailed; as = d->tex.levels; break;
            case 8: type = T3D; bl = d->tex.most_detailed; lv = d->tex.levels; break;
            case 9: type = TCUBE; bl = d->tex.most_detailed; lv = d->tex.levels; break;
            case 10: type = TCUBE; bl = d->tex.most_detailed; lv = d->tex.levels; ba = d->tex.first_array; as = d->tex.array_size * 6; break;
            default: break;
        }
    } else {
        type = r.desc.dimension == resource::TEX1D ? T1D : r.desc.dimension == resource::TEX3D ? T3D
             : r.desc.depth_or_array > 1 ? T2DA : r.desc.sample_count > 1 ? T2DMS : T2D;
    }
    image_words(w, r, format, type, bl, lv, ba, as, mapping);
}

struct UavDesc {
    uint32_t format, dim;
    union {
        struct { uint64_t first; uint32_t count, stride; uint64_t counter_offset; uint32_t flags; } buffer;
        struct { uint32_t mip, first_array, array_size, plane; } tex;
    };
};
static_assert(sizeof(UavDesc) == 40);

void WINAPI create_uav(void*, void* res, void*, const UavDesc* d, uint64_t handle) {
    if (!handle) return;
    auto* w = reinterpret_cast<uint32_t*>(handle);
    memset(w, 0, 32);
    resource::Info r;
    if (!res || !resource::info(res, r)) return;
    if (r.desc.dimension == resource::BUFFER) {
        if (d) buffer_view(w, r.address, d->format, d->buffer.first, d->buffer.count, d->buffer.stride, d->buffer.flags & 1);
        else buffer_view(w, r.address, 0, 0, uint32_t(r.desc.width / 4), 0, true);
        return;
    }
    uint32_t format = d && d->format ? d->format : r.desc.format;
    images::viewed(res, format);
    uint32_t type = T2D, mip = 0, ba = 0, as = 0;
    if (d) {
        mip = d->tex.mip;
        switch (d->dim) {
            case 2: type = T1D; break;
            case 3: type = T1DA; ba = d->tex.first_array; as = d->tex.array_size; break;
            case 4: type = T2D; break;
            case 5: type = T2DA; ba = d->tex.first_array; as = d->tex.array_size; break;
            case 8: type = T3D; break;
        }
    }
    image_words(w, r, format, type, mip, 1, ba, as, 0x1688);
}

// Views of memory the title placed itself (no resource object): (resource desc, view desc, handle).
// The view description is the D3D12 SRV / UAV description (10 words) followed by the address
// (words 10-11), as VIctory found.
void WINAPI create_placed_view(void*, const resource::ResourceDesc* rd, const void* vd, uint64_t handle, bool uav) {
    if (!handle || !rd || !vd) return;
    const uint32_t* v = static_cast<const uint32_t*>(vd);
    resource::Info r{};
    r.desc = *rd;
    r.address = uint64_t(v[10]) | uint64_t(v[11]) << 32;
    auto* w = reinterpret_cast<uint32_t*>(handle);
    memset(w, 0, 32);
    if (!r.address) return;
    if (rd->dimension == resource::BUFFER) {
        if (uav) {
            const auto* d = static_cast<const UavDesc*>(vd);
            buffer_view(w, r.address, d->format, d->buffer.first, d->buffer.count, d->buffer.stride, d->buffer.flags & 1);
        } else {
            const auto* d = static_cast<const SrvDesc*>(vd);
            buffer_view(w, r.address, d->format, d->buffer.first, d->buffer.count, d->buffer.stride, d->buffer.flags & 1);
        }
        return;
    }
    uint32_t format = v[0] ? v[0] : rd->format;
    images::viewed_placed(r.address, rd, format);
    uint32_t type = rd->dimension == resource::TEX1D ? T1D : rd->dimension == resource::TEX3D ? T3D
                  : rd->depth_or_array > 1 ? T2DA : T2D;
    uint32_t bl = 0, lv = uav ? 1 : ~0u, ba = 0, as = 0, mapping = 0x1688;
    if (!uav) {
        const auto* d = static_cast<const SrvDesc*>(vd);
        mapping = d->mapping ? d->mapping : 0x1688;
        switch (d->dim) {
            case 4: case 8: case 9: bl = d->tex.most_detailed, lv = d->tex.levels; break;
            case 5: case 10: bl = d->tex.most_detailed, lv = d->tex.levels, ba = d->tex.first_array, as = d->tex.array_size; break;
            default: break;
        }
        if (d->dim == 9 || d->dim == 10) type = TCUBE;
    }
    image_words(w, r, format, type, bl, lv, ba, as, mapping);
}
void WINAPI create_placed_srv(void* d, const resource::ResourceDesc* rd, const void* vd, uint64_t h) { create_placed_view(d, rd, vd, h, false); }
void WINAPI create_placed_uav(void* d, const resource::ResourceDesc* rd, const void* vd, uint64_t h) { create_placed_view(d, rd, vd, h, true); }

// Samplers: D3D12_SAMPLER_DESC -> S#.
struct SamplerDesc {
    uint32_t filter, u, v, w;
    float bias;
    uint32_t aniso, compare;
    float border[4], min_lod, max_lod;
};
uint32_t clamp_mode(uint32_t a) {  // D3D 1 wrap 2 mirror 3 clamp 4 border 5 mirror once -> SQ_TEX_*
    switch (a) { case 1: return 0; case 2: return 1; case 3: return 2; case 4: return 6; case 5: return 3; default: return 0; }
}
uint32_t lod_fixed(float f) { float c = f < 0 ? 0 : f > 15.99f ? 15.99f : f; return uint32_t(c * 256.0f) & 0xfff; }
void sampler_words(uint32_t* w, const SamplerDesc& d) {
    memset(w, 0, 16);
    bool aniso = (d.filter & 0x40) != 0, cmp = (d.filter & 0x80) != 0;
    uint32_t mip = (d.filter >> 0) & 1, mag = (d.filter >> 2) & 1, min = (d.filter >> 4) & 1;
    uint32_t ratio = 0;
    for (uint32_t a = d.aniso; a > 1 && ratio < 4; a >>= 1) ++ratio;
    w[0] = clamp_mode(d.u) | clamp_mode(d.v) << 3 | clamp_mode(d.w) << 6;
    if (aniso) w[0] |= ratio << 9;
    if (cmp) w[0] |= (d.compare ? (d.compare - 1) & 7 : 0) << 12;
    w[0] |= 1u << 27;  // TRUNC_COORD as D3D
    w[1] = lod_fixed(d.min_lod) | lod_fixed(d.max_lod) << 12;
    int bias = int(d.bias * 256.0f);
    w[2] = uint32_t(bias) & 0x3fff;
    w[2] |= (aniso ? 2u : mag) << 20 | (aniso ? 2u : min) << 22 | (mip + 1) << 26;
    // Border: transparent black / opaque black / opaque white.
    bool white = d.border[0] == 1 && d.border[3] == 1;
    bool opaque = d.border[3] == 1;
    w[3] = (white ? 2u : opaque ? 1u : 0u) << 30;
}
void WINAPI create_sampler(void*, const SamplerDesc* d, uint64_t handle) {
    if (handle && d) {
        sampler_words(reinterpret_cast<uint32_t*>(handle), *d);
        images::sampler(reinterpret_cast<uint32_t*>(handle));
    }
}
// The console's own sampler description (27 words): leading words as the PC layout for now.
void WINAPI create_sampler_x(void* dev, const SamplerDesc* d, uint64_t handle) {
    static bool once = false;
    if (!once) once = true, KL_WARN("gpu", "console sampler descriptions read as PC ones (layout not verified)");
    create_sampler(dev, d, handle);
}

// Render target / depth views: ours alone (128-byte slots): tag, resource, view description.
struct TargetRecord {
    uint32_t tag, described;
    void* resource;
    uint32_t desc[6];
};
constexpr uint32_t RTV_TAG = 0x56545276, DSV_TAG = 0x56534476;  // "vRTV", "vDSV"
void target(void* res, const uint32_t* d, uint64_t handle, uint32_t tag) {
    if (!handle) return;
    TargetRecord t{tag, d ? 1u : 0u, res, {}};
    if (d) memcpy(t.desc, d, sizeof t.desc);
    memcpy(reinterpret_cast<void*>(handle), &t, sizeof t);
}
void WINAPI create_rtv(void*, void* res, const uint32_t* d, uint64_t handle) {
    target(res, d, handle, RTV_TAG);
    images::of_resource(res);
}
void WINAPI create_dsv(void*, void* res, const uint32_t* d, uint64_t handle) {
    target(res, d, handle, DSV_TAG);
    images::of_resource(res);
}

// Copies: descriptors are bytes, as on the console.
void WINAPI copy_simple(void*, uint32_t count, uint64_t dst, uint64_t src, uint32_t kind) {
    if (dst && src) memmove(reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), size_t(count) * increment(kind));
}
void WINAPI copy(void*, uint32_t dn, const uint64_t* ds, const uint32_t* dsz, uint32_t sn, const uint64_t* ss,
                 const uint32_t* ssz, uint32_t kind) {
    uint32_t step = increment(kind);
    if (!step || !ds || !ss) return;
    uint32_t di = 0, si = 0, doff = 0, soff = 0;
    while (di < dn && si < sn) {
        uint32_t dc = dsz ? dsz[di] : 1, sc = ssz ? ssz[si] : 1;
        uint32_t n = std::min(dc - doff, sc - soff);
        memmove(reinterpret_cast<void*>(ds[di] + uint64_t(doff) * step), reinterpret_cast<void*>(ss[si] + uint64_t(soff) * step), size_t(n) * step);
        doff += n, soff += n;
        if (doff == dc) ++di, doff = 0;
        if (soff == sc) ++si, soff = 0;
    }
}


struct Init {
    Init() {
        device_set(14, (void*)&create_heap);     // 0x70
        device_set(15, (void*)&increment_size);  // 0x78
        device_set(17, (void*)&create_cbv);      // 0x88
        device_set(18, (void*)&create_srv);      // 0x90
        device_set(19, (void*)&create_uav);      // 0x98
        device_set(20, (void*)&create_rtv);      // 0xa0
        device_set(21, (void*)&create_dsv);      // 0xa8
        device_set(22, (void*)&create_sampler);  // 0xb0
        device_set(23, (void*)&copy);            // 0xb8
        device_set(24, (void*)&copy_simple);     // 0xc0
        device_set(61, (void*)&create_sampler_x);   // 0x1e8
        device_set(70, (void*)&create_placed_srv);  // 0x230
        device_set(71, (void*)&create_placed_uav);  // 0x238
    }
} g_init;

}  // namespace

bool heap_info(void* obj, uint32_t& kind, uint64_t& start) {
    auto* h = static_cast<Heap*>(obj);
    if (!h || h->object.functions != heap_table()) return false;
    kind = h->desc.kind, start = h->start;
    return true;
}

bool target_info(uint64_t handle, Target& out) {
    if (!handle) return false;
    TargetRecord t;
    memcpy(&t, reinterpret_cast<const void*>(handle), sizeof t);
    if (t.tag != RTV_TAG && t.tag != DSV_TAG) return false;
    out = {};
    out.resource = t.resource;
    out.depth = t.tag == DSV_TAG;
    if (!t.described) return true;
    out.format = t.desc[0];
    uint32_t dim = t.desc[1];
    if (!out.depth) {
        // RTV: 2 1D, 3 1D array, 4 2D, 5 2D array, 6 MS, 7 MS array, 8 3D.
        out.mip = (dim == 6 || dim == 7) ? 0 : t.desc[2];
        if (dim == 3 || dim == 5 || dim == 8) out.layer = t.desc[3];
        if (dim == 7) out.layer = t.desc[2];
    } else {
        // DSV: {format, dim, flags, ...}: 1 1D, 2 1D array, 3 2D, 4 2D array, 5 MS, 6 MS array.
        out.mip = (dim == 5 || dim == 6) ? 0 : t.desc[3];
        if (dim == 2 || dim == 4) out.layer = t.desc[4];
        if (dim == 6) out.layer = t.desc[3];
    }
    return true;
}

void sampler_from_static(const uint32_t* desc13, uint32_t* words) {
    // D3D12_STATIC_SAMPLER_DESC: filter, u, v, w, bias, aniso, compare, border colour (enum), min, max lod, ...
    SamplerDesc d{};
    memcpy(&d.filter, desc13, 4 * 7);
    uint32_t border = desc13[7];
    d.border[3] = border == 0 ? 0.0f : 1.0f;
    d.border[0] = d.border[1] = d.border[2] = border == 2 ? 1.0f : 0.0f;
    memcpy(&d.min_lod, desc13 + 8, 8);
    sampler_words(words, d);
}

uint32_t increment(uint32_t kind) {
    static const uint32_t sizes[] = {32, 16, 128, 128};
    return kind < 4 ? sizes[kind] : 0;
}

}  // namespace kl::gpu::descriptor
