// keylime GPU: command allocators and lists.
//
// The title's inlined code writes GPU packets into a list only while its put
// pointer is below a limit, and calls the list's method otherwise. Our lists
// keep the pointer and every limit at zero, so every command arrives as a call.
//
// keylime's design: a call only appends a compact command to the list's own
// stream (title threads return at once). Translation to Vulkan happens at
// submission, on the queue's submission thread, in parallel with the title.
//
// Layout and slot map reverse engineered by VIctory (list.rs).
#include "gpu/list.h"

#include "gpu/device.h"
#include "gpu/fence.h"
#include "gpu/stream.h"
#include "log.h"

namespace kl::gpu::list {
namespace {

const Iid ALLOCATOR_IID = guid(0x6102dee4, 0xaf59, 0x4b09, {0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24});
const Iid LIST_IIDS[8] = {
    guid(0x7116d91c, 0xe7e4, 0x47ce, {0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5}),
    guid(0x5b160d0f, 0xac1b, 0x4185, {0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55}),
    guid(0x553103fb, 0x1fe7, 0x4557, {0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7}),
    guid(0x38c3e585, 0xff17, 0x412c, {0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28}),
    guid(0x6fda83a7, 0xb84c, 0x4e38, {0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d}),
    guid(0x8754318e, 0xd3a9, 0x4541, {0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74}),
    guid(0x55050859, 0x4024, 0x474c, {0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74}),
    guid(0xc3827890, 0x4024, 0x474c, {0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74}),
};

bool is_pageable(const Iid* iid) {
    for (auto& p : fence::PAGEABLE_IIDS)
        if (same(iid, p)) return true;
    return false;
}

struct Allocator {
    Object object;
    uint32_t kind;
};

// The layout the title's inlined code reads; ours follows it.
struct List {
    Object object;
    uint32_t kind;
    uint32_t pad;
    uint64_t put[5];  // put pointer + plain/draw/API/unbiased-API limits: all zero
    uint32_t reserved[64];
    uint32_t graphics_root_headers[128];
    uint32_t compute_root_headers[128];
    uint32_t pipeline_shadow[8];
    uint32_t newer[256];  // newer titles' lists run past 0x560 (state flush writes +0x544..)
    // ---- ours ----
    Stream stream;
    Allocator* allocator;
};
static_assert(offsetof(List, put) == 0x18);
static_assert(offsetof(List, pipeline_shadow) + 32 == 0x560);

void** allocator_table();
void** list_table(bool newer);

HRESULT WINAPI allocator_query(Allocator* a, const Iid* iid, void** out) {
    return answer(Kind::Allocator, iid, (same(iid, ALLOCATOR_IID) || is_pageable(iid)) ? a : nullptr, out);
}
HRESULT WINAPI allocator_reset(Allocator*) { return S_OK; }

HRESULT WINAPI create_allocator(void*, uint32_t kind, const Iid* iid, void** out) {
    auto* a = new Allocator{{allocator_table(), 0, {0}}, kind};
    return allocator_query(a, iid, out);
}

bool is_list(const Iid* iid) {
    for (auto& l : LIST_IIDS)
        if (same(iid, l)) return true;
    return same(iid, fence::PAGEABLE_IIDS[1]) || same(iid, fence::PAGEABLE_IIDS[2]);
}

HRESULT new_list(uint32_t kind, void* allocator, const Iid* iid, void** out) {
    if (!allocator) return E_INVALIDARG;
    auto* l = new List();
    l->object.functions = list_table(newer_layouts());
    l->kind = kind;
    l->allocator = static_cast<Allocator*>(allocator);
    l->stream.open(kind);
    return answer(Kind::List, iid, is_list(iid) ? l : nullptr, out);
}

HRESULT WINAPI create_list(void*, uint32_t, uint32_t kind, void* allocator, void*, const Iid* iid, void** out) {
    return new_list(kind, allocator, iid, out);
}
struct ListDesc {
    uint32_t kind, flags, nodes, barrier_batch;
};
HRESULT WINAPI create_list_x(void*, const ListDesc* d, void* allocator, void*, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    return new_list(d->kind, allocator, iid, out);
}

HRESULT WINAPI list_query(List* l, const Iid* iid, void** out) { return answer(Kind::List, iid, is_list(iid) ? l : nullptr, out); }
uint32_t WINAPI list_type(List* l) { return l->kind; }
HRESULT WINAPI close(List* l) {
    l->stream.close();
    return S_OK;
}
HRESULT WINAPI reset(List* l, void* allocator, void* pipeline) {
    if (!allocator) return E_INVALIDARG;
    l->allocator = static_cast<Allocator*>(allocator);
    l->stream.open(l->kind);
    if (pipeline) l->stream.set_pipeline(pipeline);
    return S_OK;
}
HRESULT WINAPI reset_x(List* l, void* allocator, void* pipeline) { return reset(l, allocator, pipeline); }
void WINAPI ignored(List*) {}
// Pipeline syncs at the top/bottom of the pipe (0x408/0x410): a full barrier for now.
struct SyncArgs { uint64_t a, b; };
void WINAPI sync(List* l, uint64_t a, uint64_t b) { l->stream.add(Op::Barrier, SyncArgs{a, b}); }

void** allocator_table() {
    static void** t = [] {
        void** t = make_table(Kind::Allocator, 24);
        fill_object(t, (void*)&allocator_query);
        fill_child(t);
        t[8] = (void*)&allocator_reset;  // 0x40
        t[9] = (void*)&ignore;           // 0x48 allocation callback
        return t;
    }();
    return t;
}

// Slots in the older layout; newer lists have every slot from 59 (0x1d8) two later.
void** build(bool newer) {
    void** t = make_table(Kind::List, kMaxSlots);
    fill_object(t, (void*)&list_query);
    fill_child(t);
    auto at = [&](size_t slot, void* fn) { t[newer && slot >= 59 ? slot + 2 : slot] = fn; };
    at(8, (void*)&list_type);   // 0x40
    at(9, (void*)&close);       // 0x48
    at(10, (void*)&reset);      // 0x50
    at(56, (void*)&ignored);    // 0x1c0 SetMarker
    at(57, (void*)&ignored);    // 0x1c8 BeginEvent
    at(58, (void*)&ignored);    // 0x1d0 EndEvent
    at(73, (void*)&ignored);    // 0x248 pixel shader depth order
    at(99, (void*)&ignored);    // 0x318 driver hint
    at(124, (void*)&ignored);   // 0x3e0 shading rate
    at(129, (void*)&sync);      // 0x408
    at(130, (void*)&sync);      // 0x410
    at(137, (void*)&reset_x);   // 0x448
    stream::fill_list_methods(t, newer);
    return t;
}

void** list_table(bool newer) {
    static void** older = build(false);
    static void** newer_t = build(true);
    return newer ? newer_t : older;
}

struct Init {
    Init() {
        device_set(9, (void*)&create_allocator);  // 0x48
        device_set(12, (void*)&create_list);      // 0x60
        device_set(73, (void*)&create_list_x);    // 0x248
    }
} g_init;

}  // namespace

Stream* stream_of(void* list) {
    auto* l = static_cast<List*>(list);
    if (!l || (l->object.functions != list_table(false) && l->object.functions != list_table(true))) return nullptr;
    return &l->stream;
}

VkCommandBuffer close_for_submit(void*) { return VK_NULL_HANDLE; }
void submitted(void*, VkQueue) {}

}  // namespace kl::gpu::list
