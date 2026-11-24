// keylime GPU: descriptor heaps. Handles are plain addresses (CPU and GPU
// alike, as on the console). Sizes: views 32 bytes and samplers 16 (RDNA 2's
// image and sampler descriptor sizes, so a title copying descriptors as bytes
// moves whole ones); render target / depth views 128 (ours alone).
//
// keylime's plan (differs from VIctory): descriptor memory will hold real
// RDNA 2 descriptor words, which translated shaders read natively through
// raw addresses -- no mirroring of title heaps into host heaps every frame.
#include <windows.h>

#include "gpu/device.h"
#include "gpu/descriptor.h"
#include "gpu/fence.h"
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
        // Below 2^40 like all GPU-visible title memory.
        void* m = VirtualAlloc(nullptr, size_t(size), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!m) return E_OUTOFMEMORY;
        h->start = uint64_t(m);
    }
    KL_TRACE("gpu", "descriptor heap: type %u, %u descriptors, flags %u, %s at %llx", d->kind, d->count, d->flags,
             placed ? "title memory" : "ours", (unsigned long long)h->start);
    return query(h, iid, out);
}

uint32_t WINAPI increment_size(void*, uint32_t kind) { return increment(kind); }

struct Init {
    Init() {
        device_set(14, (void*)&create_heap);     // 0x70
        device_set(15, (void*)&increment_size);  // 0x78
    }
} g_init;

}  // namespace

uint32_t increment(uint32_t kind) {
    static const uint32_t sizes[] = {32, 16, 128, 128};
    return kind < 4 ? sizes[kind] : 0;
}

}  // namespace kl::gpu::descriptor
