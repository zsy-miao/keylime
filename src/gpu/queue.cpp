// keylime GPU: command queues. Unlike VIctory (which waits for every
// submission), keylime submits and returns: queue signals and waits become
// Vulkan timeline semaphore operations on the GPU, and only a fence the
// title reads is mirrored back by the completion thread.
//
// Slot map (VIctory queue.rs): IUnknown 3, object 4, child 1, then
// ExecuteCommandLists 0x50, markers 0x58-0x68, Signal 0x70, Wait 0x78,
// GetTimestampFrequency 0x80, GetDesc 0xd0, page-mapping batch 0xe8,
// PresentX 0xf0, Signal2X 0x100, ExecuteCommandListsX 0x108. Newer titles
// (device created as ID3D12Device11) have every slot from 0x70 two later.
#include <mutex>
#include <vector>

#include "gpu/device.h"
#include "gpu/exec.h"
#include "gpu/fence.h"
#include "gpu/host.h"
#include "gpu/list.h"
#include "gpu/present.h"
#include "gpu/stream.h"
#include "hle/xmem.h"
#include <atomic>
#include "log.h"
#include "stats.h"

namespace kl::gpu::queue {
namespace {

const Iid QUEUE_IID = guid(0x0ec870a6, 0x5d7e, 0x4c22, {0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed});

struct QueueDesc {
    uint32_t kind;  // 0 direct, 2 compute, 3 copy
    int32_t priority;
    uint32_t flags, nodes;
};
struct QueueDescX {
    uint32_t kind, flags, engine_or_pipe, queue;
};

struct Queue {
    Object object;
    uint32_t kind;  // the title reads the list type right after the header
    QueueDesc desc;
    int host;       // exec queue
};
static_assert(offsetof(Queue, kind) == 0x10);

bool is_queue(const Iid* iid) { return same(iid, QUEUE_IID) || same(iid, fence::PAGEABLE_IIDS[0]) ||
                                       same(iid, fence::PAGEABLE_IIDS[1]) || same(iid, fence::PAGEABLE_IIDS[2]); }

HRESULT WINAPI query(Queue* q, const Iid* iid, void** out) { return answer(Kind::Queue, iid, is_queue(iid) ? q : nullptr, out); }

void WINAPI execute(Queue* q, uint32_t count, void* const* lists) {
    if (!lists) return;
    stats::add(stats::Submits);
    stats::add(stats::Lists, count);
    exec::execute(q->host, count, lists);
}
void WINAPI execute_x(Queue* q, uint32_t count, void* const* lists, uint64_t, uint32_t) { execute(q, count, lists); }
void WINAPI marker(Queue*) {}

HRESULT WINAPI signal(Queue* q, void* fobj, uint64_t value) {
    auto* f = fence::from(fobj);
    if (!f) return E_INVALIDARG;
    exec::signal(q->host, fence::semaphore(f), value);
    fence::watch(f, value);
    return S_OK;
}
HRESULT WINAPI signal2_x(Queue* q, void* f, uint64_t value, const uint8_t*, const uint8_t*) { return signal(q, f, value); }
HRESULT WINAPI wait(Queue* q, void* fobj, uint64_t value) {
    auto* f = fence::from(fobj);
    if (!f) return E_INVALIDARG;
    if (fence::completed(f) >= value) return S_OK;
    exec::wait(q->host, fence::semaphore(f), value);
    return S_OK;
}
HRESULT WINAPI timestamp_frequency(Queue*, uint64_t* out) {
    if (!out) return E_INVALIDARG;
    auto& p = host::get().props;
    *out = uint64_t(1e9 / double(p.limits.timestampPeriod));
    return S_OK;
}
QueueDesc* WINAPI get_desc(Queue* q, QueueDesc* out) {
    if (out) *out = q->desc;
    return out;
}
HRESULT WINAPI present_x(Queue* q, uint32_t planes, const void* plane_list, const void* params) {
    return present::present(q->host, planes, plane_list, params);
}

// CopyPageMappingsBatchX(count, batches, pool, pool pages, flags): maps pages of a pool (memory the
// title mapped physical pages into) at the batches' addresses. GTA VI streams textures into pool
// pages and maps them into its tiled heap this way. Layout and semantics as VIctory found them
// (queue.rs). Done at once: earlier work on the GPU sees the old mapping only if it already ran,
// which is how titles use it (after their fences).
struct PageRange { uint32_t kind, start, count; };  // kind: 0 pool pages from start, 1 page start each, 2 unmap, 3 keep
struct PageBatch { uint64_t destination; uint32_t count; const PageRange* ranges; };
HRESULT WINAPI copy_page_mappings(Queue*, uint32_t count, const PageBatch* batches, uint64_t pool, uint32_t pool_pages, uint32_t) {
    constexpr uint64_t PAGE = 64 << 10;
    if (!batches && count) return E_INVALIDARG;
    bool ok = true;
    for (uint32_t i = 0; i < count; ++i) {
        const PageBatch& b = batches[i];
        if (!b.ranges) continue;
        uint64_t at = b.destination;
        for (uint32_t k = 0; k < b.count; ++k) {
            const PageRange& r = b.ranges[k];
            uint64_t len = uint64_t(r.count) * PAGE, src = pool + uint64_t(r.start) * PAGE;
            auto inside = [&](uint64_t pages) { return uint64_t(r.start) + pages <= pool_pages; };
            switch (r.kind) {
                case 0: ok &= inside(r.count) && xmem::map_alias(at, src, len); break;
                case 1:
                    if (!inside(1)) ok = false;
                    else
                        for (uint32_t p = 0; p < r.count; ++p) ok &= xmem::map_alias(at + p * PAGE, src, PAGE);
                    break;
                case 2: ok &= xmem::map_alias(at, 0, len); break;
                case 3: break;
                default: ok = false;
            }
            at += len;
        }
    }
    static std::atomic<int> warned{0};
    if (!ok && warned++ < 4) KL_WARN("gpu", "a page mapping into the tiled heap failed (pool %llx)", (unsigned long long)pool);
    return S_OK;
}

void** make(bool newer) {
    void** t = make_table(Kind::Queue, 40);
    fill_object(t, (void*)&query);
    fill_child(t);
    auto at = [&](size_t slot, void* fn) { t[newer && slot >= 14 ? slot + 2 : slot] = fn; };
    at(10, (void*)&execute);              // 0x50
    at(11, (void*)&marker);               // 0x58
    at(12, (void*)&marker);               // 0x60
    at(13, (void*)&marker);               // 0x68
    at(14, (void*)&signal);               // 0x70 (0x80)
    at(15, (void*)&wait);                 // 0x78 (0x88)
    at(16, (void*)&timestamp_frequency);  // 0x80 (0x90)
    at(26, (void*)&get_desc);             // 0xd0
    at(29, (void*)&copy_page_mappings);   // 0xe8 (0xf8)
    at(30, (void*)&present_x);            // 0xf0 (0x100)
    at(32, (void*)&signal2_x);            // 0x100
    at(33, (void*)&execute_x);            // 0x108 (0x118)
    return t;
}

HRESULT WINAPI create(void*, const QueueDesc* d, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    static void** older = make(false);
    static void** newer = make(true);
    auto* q = new Queue();
    q->object.functions = newer_layouts() ? newer : older;
    q->kind = d->kind;
    q->desc = *d;
    q->host = exec::queue_for(d->kind);
    KL_INFO("gpu", "command queue: type %u, priority %d", d->kind, d->priority);
    if (!is_queue(iid)) return answer(Kind::Queue, iid, nullptr, out);
    return answer(Kind::Queue, iid, q, out);
}

HRESULT WINAPI create_x(void* dev, const QueueDescX* x, const Iid* iid, void** out) {
    if (!x) return E_INVALIDARG;
    KL_INFO("gpu", "console command queue: type %u, engine/pipe %u, queue %u", x->kind, x->engine_or_pipe, x->queue);
    QueueDesc d{x->kind, 0, x->flags, 1};
    return create(dev, &d, iid, out);
}

struct Init {
    Init() {
        device_set(8, (void*)&create);     // 0x40
        device_set(48, (void*)&create_x);  // 0x180
        device_set(36, (void*)&fence::create);  // 0x120
    }
} g_init;

}  // namespace
}  // namespace kl::gpu::queue
