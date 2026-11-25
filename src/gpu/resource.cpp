// keylime GPU: heaps and resources. Semantics and slots after VIctory
// (resource.rs); memory placement is keylime's own: heaps come from memory
// below 2^40 (the console's GPU address range, which the ray-tracing builder
// insists on), committed resources from the title's own allocator (the title
// hooks it to place graphics memory itself), and texture sizes from AMD's
// addrlib.
#include "gpu/resource.h"

#include <mutex>
#include <unordered_map>

#include "gpu/device.h"
#include "gpu/fence.h"
#include "gpu/memory.h"
#include "hle/xmem.h"
#include "log.h"
#include "stats.h"

namespace kl::gpu::resource {
namespace {

constexpr uint64_t KIB64 = 64 << 10;
const Iid HEAP_IID = guid(0x6b3b2502, 0x6e51, 0x45b3, {0x90, 0xee, 0x98, 0x84, 0x26, 0x5e, 0x8d, 0xf3});
const Iid RESOURCE_IID = guid(0x696442be, 0xa72e, 0x4059, {0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad});

bool pageable(const Iid* iid) {
    for (auto& p : fence::PAGEABLE_IIDS)
        if (same(iid, p)) return true;
    return false;
}

struct HeapDesc {
    uint64_t size;
    HeapProperties properties;
    uint64_t alignment;
    uint32_t flags;
};
static_assert(sizeof(HeapDesc) == 48);

struct AllocationInfo {
    uint64_t size, alignment;
};

struct Heap {
    Object object;
    HeapDesc desc;
    uint8_t* memory;
};

enum class Backing : uint8_t { Committed, Placed, Address };

struct Resource {
    Object object;
    ResourceDesc desc;
    HeapProperties heap;
    uint32_t heap_flags;
    Backing backing;
    uint8_t* data;
    Heap* placed_in;
    uint64_t attributes;  // title-allocator attributes of committed memory (0: ours)
};

void** heap_table();
void** resource_table();

std::mutex g_lock;
std::unordered_map<uint64_t, Resource*> g_by_address;  // first resource at each address (diagnostics/ownership)

// ------------------------------------------------------------- sizes ----

uint32_t mip_count(const ResourceDesc& d) {
    if (d.mip_levels) return d.mip_levels;
    return texture::full_mip_count(uint32_t(d.width), d.height, d.dimension == TEX3D ? d.depth_or_array : 1);
}

bool footprint(const ResourceDesc& d, AllocationInfo& out) {
    if (d.dimension == BUFFER) {
        if (!d.width) return false;
        out.alignment = d.alignment > KIB64 ? d.alignment : KIB64;
        out.size = (d.width + out.alignment - 1) & ~(out.alignment - 1);
        return true;
    }
    texture::Layout l;
    if (texture_layout(d, l)) {
        // As the console: the layout's own alignment unless more is asked (4 KiB for 4 KiB
        // modes). The title keeps these sizes in its texture tables and asserts on mismatch.
        uint64_t a = d.alignment > l.alignment ? d.alignment : l.alignment;
        if (!a || (a & (a - 1))) return false;
        out.alignment = a;
        out.size = (l.size + a - 1) & ~(a - 1);
        return true;
    }
    KL_WARN("gpu", "no layout for texture: dim %u, %llux%u, format %u, layout %x", d.dimension,
            (unsigned long long)d.width, d.height, d.format, d.layout);
    return false;
}

// Resources placed one after another: the total footprint.
AllocationInfo place(const ResourceDesc* descs, uint32_t count, AllocationInfo* each_offsets = nullptr) {
    AllocationInfo total{0, 1};
    for (uint32_t i = 0; i < count; ++i) {
        AllocationInfo f;
        if (!footprint(descs[i], f)) return {~0ull, 1};
        uint64_t at = (total.size + f.alignment - 1) & ~(f.alignment - 1);
        if (each_offsets) each_offsets[i] = {at, f.alignment};
        total.size = at + f.size;
        if (f.alignment > total.alignment) total.alignment = f.alignment;
    }
    return total;
}

// Title-allocator attributes for resource memory: graphics allocator id, page size
// (2 MiB pages from 2 MiB up), log2 alignment, memory type (write-combined for GPU-only).
uint64_t resource_attributes(const AllocationInfo& info, uint32_t heap_kind) {
    const uint64_t GRAPHICS_ALLOCATOR = 128ull << 16;
    uint64_t pages = info.size >= (2u << 20) ? 2 : 1;
    uint64_t memory = heap_kind == 1 ? 1 : 3;
    unsigned long a = 0;
    _BitScanForward64(&a, info.alignment ? info.alignment : 1);
    return GRAPHICS_ALLOCATOR | pages << 14 | uint64_t(a) << 24 | memory << 29;
}

// ---------------------------------------------------------- creation ----

HRESULT WINAPI resource_query(Resource* r, const Iid* iid, void** out) {
    return answer(Kind::Resource, iid, (same(iid, RESOURCE_IID) || pageable(iid)) ? r : nullptr, out);
}

HRESULT hand_out(const ResourceDesc& d, const HeapProperties& heap, uint32_t heap_flags, Backing b, uint8_t* data,
                 Heap* in, uint64_t attributes, const Iid* iid, void** out) {
    stats::add(stats::Resources);
    auto* r = new Resource{{resource_table(), 0, {0}}, d, heap, heap_flags, b, data, in, attributes};
    {
        std::lock_guard g(g_lock);
        g_by_address.try_emplace(uint64_t(data), r);
    }
    // The GPU sees the memory as it is (imported, no copy).
    AllocationInfo f;
    if (data && footprint(d, f)) memory::ensure(uint64_t(data), f.size);
    KL_TRACE("gpu", "resource %s dim %u %llux%u fmt %u at %p", b == Backing::Committed ? "committed" : b == Backing::Placed ? "placed" : "at address",
             d.dimension, (unsigned long long)d.width, d.height, d.format, (void*)data);
    return resource_query(r, iid, out);
}

HRESULT WINAPI create_committed(void*, const HeapProperties* heap, uint32_t heap_flags, const ResourceDesc* d,
                                uint32_t, const void*, const Iid* iid, void** out) {
    if (!heap || !d) return E_INVALIDARG;
    AllocationInfo f;
    if (!footprint(*d, f)) return E_INVALIDARG;
    uint64_t attr = resource_attributes(f, heap->kind);
    auto* data = static_cast<uint8_t*>(xmem::alloc(size_t(f.size), attr));
    if (!data) return E_OUTOFMEMORY;
    memset(data, 0, size_t(f.size));
    return hand_out(*d, *heap, heap_flags, Backing::Committed, data, nullptr, attr, iid, out);
}

HRESULT WINAPI create_placed(void*, Heap* heap, uint64_t offset, const ResourceDesc* d, uint32_t, const void*,
                             const Iid* iid, void** out) {
    if (!d || !heap) return E_INVALIDARG;
    AllocationInfo f;
    if (!footprint(*d, f) || offset % f.alignment || offset + f.size > heap->desc.size) {
        KL_WARN("gpu", "placed resource does not fit: offset %llu, heap %llu", (unsigned long long)offset,
                (unsigned long long)heap->desc.size);
        return E_INVALIDARG;
    }
    ++heap->object.refs;
    return hand_out(*d, heap->desc.properties, heap->desc.flags, Backing::Placed, heap->memory + offset, heap, 0, iid, out);
}

// Device 0x3f8: enhanced-barrier committed creation (heap, flags, desc1, layout, clear, protected
// session, castable count, castable formats, iid, out); the title's back buffers come this way.
HRESULT WINAPI create_committed3(void* dev, const HeapProperties* heap, uint32_t flags, const ResourceDesc* d, uint32_t,
                                 const void* clear, void*, uint32_t, const uint32_t*, const Iid* iid, void** out) {
    return create_committed(dev, heap, flags, d, 0, clear, iid, out);
}

// Device 0x428 (Device9..11 addition): a resource at a virtual address the title mapped
// itself, (address, desc1, initial layout, clear, iid, out). GPU addresses are CPU addresses.
HRESULT WINAPI create_at_address(void*, uint64_t address, const ResourceDesc* d, uint32_t, const void*,
                                 const Iid* iid, void** out) {
    if (!d || !address) return E_INVALIDARG;
    HeapProperties dflt{1, 0, 0, 1, 1};
    return hand_out(*d, dflt, 0, Backing::Address, reinterpret_cast<uint8_t*>(address), nullptr, 0, iid, out);
}

HRESULT WINAPI heap_query(Heap* h, const Iid* iid, void** out) {
    return answer(Kind::Heap, iid, (same(iid, HEAP_IID) || pageable(iid)) ? h : nullptr, out);
}
void WINAPI heap_desc(Heap* h, HeapDesc* out) { if (out) *out = h->desc; }

HRESULT WINAPI create_heap(void*, const HeapDesc* d, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    if (d->alignment && (d->alignment & (d->alignment - 1))) return E_INVALIDARG;
    auto* mem = static_cast<uint8_t*>(xmem::gpu_alloc(size_t(d->size), size_t(d->alignment > KIB64 ? d->alignment : KIB64)));
    if (!mem) {
        KL_WARN("gpu", "heap of %llu bytes: out of memory", (unsigned long long)d->size);
        return E_OUTOFMEMORY;
    }
    KL_TRACE("gpu", "heap %llu bytes, type %u, flags %x at %p", (unsigned long long)d->size, d->properties.kind, d->flags, (void*)mem);
    memory::ensure(uint64_t(mem), d->size);
    auto* h = new Heap{{heap_table(), 0, {0}}, *d, mem};
    return heap_query(h, iid, out);
}

// Device GetResourceAllocationInfo: returns its 16-byte struct through a hidden first argument.
AllocationInfo* WINAPI allocation_info(AllocationInfo* out, void*, uint32_t, uint32_t count, const ResourceDesc* d) {
    *out = d ? place(d, count) : AllocationInfo{0, 1};
    return out;
}
struct ResourceDesc1 {
    ResourceDesc desc;
    uint32_t mip_region[3];
};
static_assert(sizeof(ResourceDesc1) == 72);
struct AllocationInfo1 {
    uint64_t offset, alignment, size;
};
AllocationInfo* WINAPI allocation_info2(AllocationInfo* out, void*, uint32_t, uint32_t count, const ResourceDesc1* d,
                                        AllocationInfo1* each) {
    AllocationInfo total{0, 1};
    for (uint32_t i = 0; d && i < count; ++i) {
        AllocationInfo f;
        if (!footprint(d[i].desc, f)) {
            total = {~0ull, 1};
            break;
        }
        uint64_t at = (total.size + f.alignment - 1) & ~(f.alignment - 1);
        if (each) each[i] = {at, f.alignment, f.size};
        total.size = at + f.size;
        if (f.alignment > total.alignment) total.alignment = f.alignment;
    }
    *out = total;
    return out;
}

HeapProperties* WINAPI custom_heap_properties(HeapProperties* out, void*, uint32_t nodes, uint32_t kind) {
    if (!nodes) nodes = 1;
    uint32_t cpu = kind == 1 ? 1u : (kind == 2 || kind == 3) ? 3u : 0u;
    if (out) *out = {4, cpu, cpu ? 1u : 0u, nodes, nodes};
    return out;
}

// Page pools: memory the title maps pages into (tiled resources); remembered, used by page mappings.
std::mutex g_pool_lock;
std::vector<std::pair<uint64_t, uint32_t>> g_pools;
HRESULT WINAPI register_page_pool(void*, uint64_t address, uint32_t pages, size_t* out) {
    if (!out || !pages || address % KIB64) return E_INVALIDARG;
    std::lock_guard g(g_pool_lock);
    g_pools.push_back({address, pages});
    *out = g_pools.size();
    return S_OK;
}
HRESULT WINAPI unregister_page_pool(void*, size_t handle) {
    std::lock_guard g(g_pool_lock);
    if (!handle || handle > g_pools.size()) return E_INVALIDARG;
    g_pools[handle - 1] = {0, 0};
    return S_OK;
}

// ---------------------------------------------------- resource methods ----

uint32_t WINAPI resource_release(Resource* r) {
    uint32_t left = --r->object.refs;
    // Console memory outlives the object, and lists may still name it: objects stay (for now).
    return left;
}
HRESULT WINAPI map(Resource* r, uint32_t, const void*, void** out) {
    if (out) *out = r->data;
    return S_OK;
}
void WINAPI unmap(Resource*, uint32_t, const void*) {}
void WINAPI resource_desc(Resource* r, ResourceDesc* out) { if (out) *out = r->desc; }
uint64_t WINAPI gpu_address(Resource* r) { return uint64_t(r->data); }
HRESULT WINAPI heap_properties(Resource* r, HeapProperties* p, uint32_t* flags) {
    if (p) *p = r->heap;
    if (flags) *flags = r->heap_flags;
    return S_OK;
}

void** heap_table() {
    static void** t = [] {
        void** t = make_table(Kind::Heap, 24);
        fill_object(t, (void*)&heap_query);
        fill_child(t);
        t[8] = (void*)&heap_desc;  // 0x40
        return t;
    }();
    return t;
}
void** resource_table() {
    static void** t = [] {
        void** t = make_table(Kind::Resource, 32);
        fill_object(t, (void*)&resource_query, (void*)&resource_release);
        fill_child(t);
        t[8] = (void*)&map;               // 0x40
        t[9] = (void*)&unmap;             // 0x48
        t[10] = (void*)&resource_desc;    // 0x50
        t[11] = (void*)&gpu_address;      // 0x58
        t[14] = (void*)&heap_properties;  // 0x70
        return t;
    }();
    return t;
}

struct Init {
    Init() {
        device_set(25, (void*)&allocation_info);         // 0xc8
        device_set(26, (void*)&custom_heap_properties);  // 0xd0
        device_set(27, (void*)&create_committed);        // 0xd8
        device_set(28, (void*)&create_heap);             // 0xe0
        device_set(29, (void*)&create_placed);           // 0xe8
        device_set(74, (void*)&register_page_pool);      // 0x250
        device_set(75, (void*)&unregister_page_pool);    // 0x258
        device_set(76, (void*)&create_committed);        // 0x260 CreateCommittedResourceX (PC desc + DCC options)
        device_set(108, (void*)&allocation_info2);       // 0x360
        device_set(127, (void*)&create_committed3);      // 0x3f8
        device_set(133, (void*)&create_at_address);      // 0x428
    }
} g_init;

}  // namespace

bool texture_layout(const ResourceDesc& d, texture::Layout& out) {
    texture::Desc t;
    switch (d.dimension) {
        case TEX1D: t.dim = texture::Dim::Tex1d; break;
        case TEX2D: t.dim = texture::Dim::Tex2d; break;
        case TEX3D: t.dim = texture::Dim::Tex3d; break;
        default: return false;
    }
    // Layout field: 0 = the library's pick, 1 = linear, 0x100|mode = that swizzle mode.
    if (d.layout == 0) t.swizzle = texture::default_swizzle(t.dim, d.flags & FLAG_DEPTH, d.flags & FLAG_RENDER_TARGET);
    else if (d.layout == 1) t.swizzle = 0;
    else if ((d.layout & ~0xffu) == 0x100) t.swizzle = d.layout & 0xff;
    else return false;
    t.width = uint32_t(d.width);
    t.height = d.height;
    t.depth_or_array = d.depth_or_array;
    t.mips = mip_count(d);
    t.format = d.format;
    t.samples = d.sample_count ? d.sample_count : 1;
    return texture::layout(t, out);
}

bool info(void* obj, Info& out) {
    auto* r = static_cast<Resource*>(obj);
    if (!r || r->object.functions != resource_table()) return false;
    out = {uint64_t(r->data), r->desc, r->heap.kind};
    return true;
}

}  // namespace kl::gpu::resource
