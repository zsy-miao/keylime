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
#include "gpu/fence.h"
#include "gpu/host.h"
#include "gpu/list.h"
#include "gpu/present.h"
#include "gpu/stream.h"
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

// One lock per host VkQueue (several title queues may share one).
struct HostQueue {
    VkQueue q;
    std::mutex lock;
};
HostQueue g_host_queues[3];

struct Queue {
    Object object;
    uint32_t kind;  // the title reads the list type right after the header
    QueueDesc desc;
    HostQueue* host;
};
static_assert(offsetof(Queue, kind) == 0x10);

bool is_queue(const Iid* iid) { return same(iid, QUEUE_IID) || same(iid, fence::PAGEABLE_IIDS[0]) ||
                                       same(iid, fence::PAGEABLE_IIDS[1]) || same(iid, fence::PAGEABLE_IIDS[2]); }

HRESULT WINAPI query(Queue* q, const Iid* iid, void** out) { return answer(Kind::Queue, iid, is_queue(iid) ? q : nullptr, out); }

void submit(Queue* q, const std::vector<VkCommandBuffer>& cbs, VkSemaphore wait, uint64_t wait_value,
            VkSemaphore signal, uint64_t signal_value) {
    std::vector<VkCommandBufferSubmitInfo> ci;
    for (auto cb : cbs) {
        VkCommandBufferSubmitInfo c{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        c.commandBuffer = cb;
        ci.push_back(c);
    }
    VkSemaphoreSubmitInfo w{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO}, s{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    w.semaphore = wait;
    w.value = wait_value;
    w.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    s.semaphore = signal;
    s.value = signal_value;
    s.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = uint32_t(ci.size());
    si.pCommandBufferInfos = ci.data();
    si.waitSemaphoreInfoCount = wait ? 1 : 0;
    si.pWaitSemaphoreInfos = &w;
    si.signalSemaphoreInfoCount = signal ? 1 : 0;
    si.pSignalSemaphoreInfos = &s;
    std::lock_guard g(q->host->lock);
    KL_VK_CHECK(vk::vkQueueSubmit2(q->host->q, 1, &si, VK_NULL_HANDLE));
}

void WINAPI execute(Queue* q, uint32_t count, void* const* lists) {
    if (!lists) return;
    std::vector<VkCommandBuffer> cbs;
    stats::add(stats::Submits);
    stats::add(stats::Lists, count);
    for (uint32_t i = 0; i < count; ++i) {
        if (Stream* s = list::stream_of(lists[i])) stream::run_cpu_side(*s);
        if (VkCommandBuffer cb = list::close_for_submit(lists[i])) cbs.push_back(cb);
    }
    if (!cbs.empty()) submit(q, cbs, VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0);
    for (uint32_t i = 0; i < count; ++i) list::submitted(lists[i], q->host->q);
}
void WINAPI execute_x(Queue* q, uint32_t count, void* const* lists, uint64_t, uint32_t) { execute(q, count, lists); }
void WINAPI marker(Queue*) {}

HRESULT WINAPI signal(Queue* q, void* fobj, uint64_t value) {
    auto* f = fence::from(fobj);
    if (!f) return E_INVALIDARG;
    submit(q, {}, VK_NULL_HANDLE, 0, fence::semaphore(f), value);
    fence::watch(f, value);
    return S_OK;
}
HRESULT WINAPI signal2_x(Queue* q, void* f, uint64_t value, const uint8_t*, const uint8_t*) { return signal(q, f, value); }
HRESULT WINAPI wait(Queue* q, void* fobj, uint64_t value) {
    auto* f = fence::from(fobj);
    if (!f) return E_INVALIDARG;
    if (fence::completed(f) >= value) return S_OK;
    submit(q, {}, fence::semaphore(f), value, VK_NULL_HANDLE, 0);
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
    return present::present(q->host->q, &q->host->lock, planes, plane_list, params);
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
    at(30, (void*)&present_x);            // 0xf0 (0x100)
    at(32, (void*)&signal2_x);            // 0x100
    at(33, (void*)&execute_x);            // 0x108 (0x118)
    return t;
}

HostQueue* host_for(uint32_t kind) {
    static std::once_flag once;
    std::call_once(once, [] {
        auto& h = host::get();
        g_host_queues[0].q = h.queues.graphics;
        g_host_queues[1].q = h.queues.compute;
        g_host_queues[2].q = h.queues.transfer;
    });
    // Copy queues go to the graphics queue for now: images there need no ownership transfers.
    return kind == 2 ? &g_host_queues[1] : &g_host_queues[0];
}

HRESULT WINAPI create(void*, const QueueDesc* d, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    static void** older = make(false);
    static void** newer = make(true);
    auto* q = new Queue();
    q->object.functions = newer_layouts() ? newer : older;
    q->kind = d->kind;
    q->desc = *d;
    q->host = host_for(d->kind);
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
