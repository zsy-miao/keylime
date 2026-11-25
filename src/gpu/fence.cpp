#include "gpu/fence.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "gpu/exec.h"
#include "gpu/host.h"
#include "log.h"

namespace kl::gpu::fence {

const Iid PAGEABLE_IIDS[3] = {
    guid(0x63ee58fb, 0x1268, 0x4835, {0x86, 0xda, 0xf0, 0x08, 0xce, 0x62, 0xf0, 0xd6}),
    guid(0x905db94b, 0xa00c, 0x4140, {0x9d, 0xf5, 0x2b, 0x64, 0xca, 0x9e, 0xa3, 0x57}),
    guid(0xc4fec28f, 0x7966, 0x4e95, {0x9f, 0x94, 0xf4, 0x31, 0xcb, 0x56, 0xc3, 0xb8}),
};
const Iid FENCE_IID = guid(0x0a753dcf, 0xc4d8, 0x4b91, {0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76});

struct Fence {
    Object object;
    std::atomic<uint64_t> value;  // the title reads this at the fence's GPU address
    uint32_t flags;
    VkSemaphore sem;
    std::mutex lock;
    std::vector<std::pair<uint64_t, HANDLE>> waiters;
    uint64_t sem_high;  // highest value the semaphore was (or will be) signalled to
};

namespace {

void** table();

bool is_fence(const Iid* iid) {
    if (same(iid, FENCE_IID)) return true;
    for (auto& p : PAGEABLE_IIDS)
        if (same(iid, p)) return true;
    return false;
}

void reached(Fence* f, uint64_t v) {
    std::lock_guard g(f->lock);
    uint64_t cur = f->value.load();
    if (v > cur) f->value.store(v, std::memory_order_release);
    std::erase_if(f->waiters, [&](auto& w) {
        if (w.first > f->value.load()) return false;
        SetEvent(w.second);
        return true;
    });
}

// ---------------------------------------------------- completion thread ----

struct Watch {
    Fence* f;
    uint64_t value;
};
std::mutex g_wlock;
std::condition_variable g_wcv;
std::vector<Watch> g_watch;

void completion_thread() {
    SetThreadDescription(GetCurrentThread(), L"keylime-gpu-completion");
    auto& h = host::get();
    for (;;) {
        std::vector<Watch> w;
        {
            std::unique_lock g(g_wlock);
            g_wcv.wait(g, [] { return !g_watch.empty(); });
            w = g_watch;
        }
        std::vector<VkSemaphore> sems;
        std::vector<uint64_t> vals;
        for (auto& x : w) sems.push_back(x.f->sem), vals.push_back(x.value);
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
        wi.semaphoreCount = uint32_t(sems.size());
        wi.pSemaphores = sems.data();
        wi.pValues = vals.data();
        VkResult r = vk::vkWaitSemaphores(h.device, &wi, 20'000'000);  // 20 ms: new watches join
        if (r == VK_ERROR_DEVICE_LOST) exec::device_lost("waiting for fences");
        std::lock_guard g(g_wlock);
        std::erase_if(g_watch, [&](Watch& x) {
            uint64_t now = 0;
            vk::vkGetSemaphoreCounterValue(h.device, x.f->sem, &now);
            if (now < x.value) return false;
            reached(x.f, now);
            return true;
        });
    }
}

// --------------------------------------------------------------- methods ----

HRESULT WINAPI query(Fence* f, const Iid* iid, void** out) { return answer(Kind::Fence, iid, is_fence(iid) ? f : nullptr, out); }
uint32_t WINAPI release(Fence* f) {
    uint32_t left = --f->object.refs;
    // Fences live on: the completion thread or a queue may still name them.
    return left;
}
uint64_t WINAPI completed_m(Fence* f) { return f->value.load(std::memory_order_acquire); }

HRESULT WINAPI set_event_on_completion(Fence* f, uint64_t value, HANDLE event) {
    if (!event) {
        // Block until reached: the GPU really runs here, so this can wait.
        uint64_t sv = value;
        VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wi.semaphoreCount = 1;
        wi.pSemaphores = &f->sem;
        wi.pValues = &sv;
        auto t0 = GetTickCount64();
        bool warned = false;
        while (f->value.load() < value) {
            VkResult r = vk::vkWaitSemaphores(host::get().device, &wi, 100'000'000);
            if (r == VK_SUCCESS) reached(f, value);
            else if (r == VK_ERROR_DEVICE_LOST) exec::device_lost("a CPU wait for a fence");
            uint64_t ms = GetTickCount64() - t0;
            if (!warned && ms > 3000) {
                warned = true;
                uint64_t now = 0;
                vk::vkGetSemaphoreCounterValue(host::get().device, f->sem, &now);
                KL_WARN("gpu", "a CPU wait for fence value %llu has taken 3 s (fence at %llu, highest signal %llu)",
                        (unsigned long long)value, (unsigned long long)now, (unsigned long long)high(f));
            }
            if (ms > 15000) {
                KL_ERROR("gpu", "a CPU wait for fence value %llu gave up after 15 s", (unsigned long long)value);
                break;
            }
        }
        return S_OK;
    }
    std::lock_guard g(f->lock);
    if (f->value.load() >= value) {
        SetEvent(event);
        return S_OK;
    }
    f->waiters.push_back({value, event});
    return S_OK;
}
HRESULT WINAPI signal_m(Fence* f, uint64_t value) {
    signal_cpu(f, value);
    return S_OK;
}
uint32_t WINAPI creation_flags(Fence* f) { return f->flags; }
uint64_t WINAPI gpu_address(Fence* f) { return reinterpret_cast<uint64_t>(&f->value); }

void** table() {
    static void** t = [] {
        void** t = make_table(Kind::Fence, 24);
        fill_object(t, (void*)&query, (void*)&release);
        fill_child(t);
        t[8] = (void*)&completed_m;               // 0x40
        t[9] = (void*)&set_event_on_completion;   // 0x48
        t[10] = (void*)&signal_m;                 // 0x50
        t[11] = (void*)&creation_flags;           // 0x58
        t[12] = (void*)&gpu_address;              // 0x60
        return t;
    }();
    return t;
}

}  // namespace

Fence* from(void* obj) {
    auto* f = static_cast<Fence*>(obj);
    return f && f->object.functions == table() ? f : nullptr;
}
VkSemaphore semaphore(Fence* f) { return f->sem; }
uint64_t completed(Fence* f) { return f->value.load(); }

bool raise(Fence* f, uint64_t value) {
    std::lock_guard g(f->lock);
    if (value <= f->sem_high) return false;
    f->sem_high = value;
    return true;
}

uint64_t high(Fence* f) {
    std::lock_guard g(f->lock);
    return f->sem_high;
}

void dump_state() {
    std::vector<Watch> w;
    {
        std::lock_guard g(g_wlock);
        w = g_watch;
    }
    KL_INFO("hang", "fences: %zu signals not reached yet", w.size());
    for (size_t i = 0; i < w.size() && i < 16; ++i) {
        uint64_t now = 0;
        vk::vkGetSemaphoreCounterValue(host::get().device, w[i].f->sem, &now);
        size_t waiters;
        {
            std::lock_guard g(w[i].f->lock);
            waiters = w[i].f->waiters.size();
        }
        KL_INFO("hang", "  fence %p: waits for %llu, at %llu, title sees %llu, %zu events waiting", (void*)w[i].f,
                (unsigned long long)w[i].value, (unsigned long long)now, (unsigned long long)w[i].f->value.load(), waiters);
    }
}

void watch(Fence* f, uint64_t value) {
    {
        std::lock_guard g(f->lock);
        if (value > f->sem_high) f->sem_high = value;
    }
    static std::once_flag once;
    std::call_once(once, [] { std::thread(completion_thread).detach(); });
    std::lock_guard g(g_wlock);
    g_watch.push_back({f, value});
    g_wcv.notify_all();
}

void signal_cpu(Fence* f, uint64_t value) {
    bool raise = false;
    {
        std::lock_guard g(f->lock);
        if (value > f->sem_high) f->sem_high = value, raise = true;
    }
    // A timeline only climbs; D3D12 lets the CPU set a lower value, which only the memory copy shows.
    if (raise) {
        VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        si.semaphore = f->sem;
        si.value = value;
        KL_VK_CHECK(vk::vkSignalSemaphore(host::get().device, &si));
    }
    std::lock_guard g(f->lock);
    f->value.store(value, std::memory_order_release);
    std::erase_if(f->waiters, [&](auto& w) {
        if (w.first > value) return false;
        SetEvent(w.second);
        return true;
    });
}

HRESULT WINAPI create(void*, uint64_t initial, uint32_t flags, const Iid* iid, void** out) {
    auto* f = new Fence();
    f->object.functions = table();
    f->value = initial;
    f->flags = flags;
    f->sem_high = initial;
    VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    ti.initialValue = initial;
    VkSemaphoreCreateInfo ci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    ci.pNext = &ti;
    if (!KL_VK_CHECK(vk::vkCreateSemaphore(host::get().device, &ci, nullptr, &f->sem))) return E_OUTOFMEMORY;
    if (!is_fence(iid)) {
        KL_WARN("gpu", "fence created as unknown interface %s", guid_text(iid).c_str());
        return E_NOINTERFACE;
    }
    return answer(Kind::Fence, iid, f, out);
}

}  // namespace kl::gpu::fence
