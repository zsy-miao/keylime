// keylime: host queues, GPU waits on memory, retiring objects, command buffers and submission (part of the command executor; included by exec.cpp).

constexpr uint64_t RECORD_BLOCK = 4ull << 20;
constexpr uint32_t RECORD_DWORDS = 64;  // VS (or CS) SGPRs at 0, PS SGPRs at 32

struct Cb {
    VkCommandBuffer cb = VK_NULL_HANDLE;
    uint64_t value = 0;
    std::vector<memory::Block> records;
    size_t block = 0;
    uint64_t used = 0;
};

struct Gate {
    cmd::WaitValue w{};
    VkSemaphore sem = VK_NULL_HANDLE;
    uint64_t sem_value = 0, opens = 0, since = 0;
};

struct HostQueue {
    VkQueue q = VK_NULL_HANDLE;
    uint32_t family = 0;
    std::mutex lock, qlock;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    std::atomic<uint64_t> submitted{0};
    // Commands are being recorded (they may name objects retired meanwhile: those wait for the
    // submission that carries them).
    std::atomic<int> recording{0};
    std::deque<Cb*> busy;
    std::vector<Cb*> idle;
    // GPU waits on title memory: work after one waits on `gate` reaching `gate_value`, which a
    // host thread signals once the memory compares as asked.
    VkSemaphore gate = VK_NULL_HANDLE;
    uint64_t gate_value = 0;
    std::mutex gates_lock;
    std::condition_variable gates_cv;
    std::deque<Gate> gates;
    std::atomic<uint64_t> opened_late{0};
    VkQueryPool stamps = VK_NULL_HANDLE;
    uint32_t stamp_next = 0;
    std::deque<uint32_t> stamps_open;
    uint64_t stamp_last = 0;
    // KEYLIME_PROFILE=1: timestamps around every draw and dispatch, charged to their pipeline (read_profile).
    struct Mark {
        uint32_t slot, kind;
        uint64_t key, list;
    };
    VkQueryPool prof = VK_NULL_HANDLE;
    uint32_t prof_next = 0;
    std::deque<Mark> prof_open;
    uint64_t prof_list = 0;
};
constexpr uint32_t STAMPS = 1024, PROF = 1 << 16;
bool profiling() {
    static const bool on = [] { char v[4]{}; return GetEnvironmentVariableA("KEYLIME_PROFILE", v, 4) && v[0] == '1'; }();
    return on;
}
HostQueue g_q[2];
VkPipelineLayout g_layout = VK_NULL_HANDLE;
VkPipelineCache g_cache = VK_NULL_HANDLE;
VkPipeline g_clamp[4] = {};  // ExecuteIndirect arguments, checked on the GPU (draw, indexed draw, dispatch, mesh dispatch)
memory::Block g_zero;  // binds where a vertex stream is missing
bool g_started = false;

VkDevice dev() { return host::get().device; }

uint64_t completed(HostQueue& h) {
    uint64_t v = 0;
    if (h.timeline) vk::vkGetSemaphoreCounterValue(dev(), h.timeline, &v);
    // NVIDIA's timelines read all ones after a GPU reset (Windows TDR) without an error code.
    if (v == ~0ull) device_lost("reading a queue's progress (all ones: the GPU was reset)");
    return v;
}
// NV checkpoints: each draw and dispatch tags the command buffer with what it runs (kind in the
// top byte, the pipeline key below), so a lost device can name the command it was on.
struct Mark {
    uint64_t key;
    uint32_t kind, x, y, z;  // group counts (0 for indirect)
};
constexpr uint32_t MARKS = 1u << 20;
Mark g_marks[MARKS];
std::atomic<uint32_t> g_mark_next{0};
void checkpoint(VkCommandBuffer cb, uint32_t kind, uint64_t key, uint32_t x = 0, uint32_t y = 0, uint32_t z = 0) {
    if (!host::get().caps.checkpoints || !vk::vkCmdSetCheckpointNV) return;
    uint32_t at = g_mark_next++ % MARKS;
    g_marks[at] = {key, kind, x, y, z};
    vk::vkCmdSetCheckpointNV(cb, reinterpret_cast<void*>(uint64_t(at) + 1));
}

// A CPU copy that survives memory the title has not mapped (yet).
bool safe_move(uint64_t dst, uint64_t src, uint64_t len) {
    __try {
        memmove(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), size_t(len));
        return true;
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

struct Event {
    uint64_t ms;
    char text[120];
};
std::mutex g_ev_lock;
Event g_ev[64];
uint64_t g_ev_n = 0;

void note(const char* fmt, ...) {
    Event e;
    e.ms = GetTickCount64();
    va_list a;
    va_start(a, fmt);
    vsnprintf(e.text, sizeof e.text, fmt, a);
    va_end(a);
    std::lock_guard g(g_ev_lock);
    g_ev[g_ev_n++ % 64] = e;
}

uint64_t read_value(const cmd::WaitValue& w, bool& ok) {
    uint64_t v = 0;
    ok = w.address && safe_move(uint64_t(&v), w.address, w.bytes == 8 ? 8 : 4);
    if (w.bytes != 8) v &= w.mask ? w.mask : 0xffffffffull;
    return v;
}

bool met(const cmd::WaitValue& w) {
    bool ok;
    uint64_t v = read_value(w, ok);
    if (!ok) return true;
    uint64_t r = w.bytes == 8 ? w.reference : (w.reference & (w.mask ? w.mask : 0xffffffffull));
    switch (w.compare) {
        case 1: return false;
        case 2: return v < r;
        case 3: return v == r;
        case 4: return v <= r;
        case 5: return v > r;
        case 6: return v != r;
        case 7: return v >= r;
        default: return true;
    }
}

bool gate_met(const Gate& g) {
    if (!g.sem) return met(g.w);
    uint64_t v = 0;
    vk::vkGetSemaphoreCounterValue(dev(), g.sem, &v);
    return v >= g.sem_value;
}

// ---------------------------------------------------------- retiring ----
struct Retiree {
    uint64_t need[2];
    std::function<void()> f;
};
std::mutex g_retire_lock;
std::deque<Retiree> g_retirees;

void collect() {
    std::vector<std::function<void()>> due;
    {
        std::lock_guard g(g_retire_lock);
        if (g_retirees.empty()) return;
        uint64_t done[2] = {completed(g_q[0]), completed(g_q[1])};
        uint64_t sub[2] = {g_q[0].submitted.load(), g_q[1].submitted.load()};
        bool rec[2] = {g_q[0].recording.load() != 0, g_q[1].recording.load() != 0};
        while (!g_retirees.empty()) {
            auto& r = g_retirees.front();
            bool ok = true;
            // Done once the submission after the retiring is done; a queue that records nothing
            // now and finished all it was given will not use the object either.
            for (int q = 0; q < 2; ++q) ok &= done[q] >= r.need[q] || (!rec[q] && sub[q] < r.need[q] && done[q] >= sub[q]);
            if (!ok) break;
            due.push_back(std::move(r.f));
            g_retirees.pop_front();
        }
    }
    for (auto& f : due) f();
}

// ----------------------------------------------------- command buffers ----
Cb* get_cb(HostQueue& h) {
    uint64_t done = completed(h);
    while (!h.busy.empty() && h.busy.front()->value <= done) {
        Cb* c = h.busy.front();
        h.busy.pop_front();
        vk::vkResetCommandBuffer(c->cb, 0);
        h.idle.push_back(c);
    }
    Cb* c;
    if (!h.idle.empty()) {
        c = h.idle.back();
        h.idle.pop_back();
    } else {
        c = new Cb();
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = h.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (!KL_VK_CHECK(vk::vkAllocateCommandBuffers(dev(), &ai, &c->cb))) {
            delete c;
            return nullptr;
        }
    }
    c->block = 0, c->used = 0;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vk::vkBeginCommandBuffer(c->cb, &bi);
    return c;
}

// A draw record of `dwords`: host pointer and device address.
uint32_t* record(Cb* c, uint32_t dwords, uint64_t& device) {
    uint64_t size = (uint64_t(dwords) * 4 + 15) & ~15ull;
    while (c->block < c->records.size() && c->used + size > c->records[c->block].size) ++c->block, c->used = 0;
    if (c->block == c->records.size()) {
        memory::Block b;
        if (!memory::make_block(RECORD_BLOCK, true, b)) return nullptr;
        c->records.push_back(b);
        c->used = 0;
    }
    auto& b = c->records[c->block];
    device = b.device + c->used;
    auto* p = reinterpret_cast<uint32_t*>(b.map + c->used);
    c->used += size;
    return p;
}

struct Semaphore {
    VkSemaphore s;
    uint64_t value;
};

struct Pending {
    HostQueue* h = nullptr;
    std::vector<VkCommandBufferSubmitInfo> ci;
    std::vector<Semaphore> waits, signals;
    uint64_t value = 0;
    bool sent = false;
};
// Work goes to the driver only once everything it waits on has been signalled or handed to the
// driver: NVIDIA's Windows driver blocks in vkQueueSubmit2 on a wait with no signal ahead of it,
// and while it blocks, vkSignalSemaphore and fence reads from other threads block too.
std::mutex g_sub_lock;
std::condition_variable g_sub_cv;
std::deque<std::shared_ptr<Pending>> g_pending;
std::unordered_map<VkSemaphore, uint64_t> g_sent;
std::atomic<size_t> g_pending_n{0};
std::atomic<uint64_t> g_held{0};

uint64_t counter(VkSemaphore s) {
    uint64_t v = 0;
    vk::vkGetSemaphoreCounterValue(dev(), s, &v);
    return v;
}

bool ready(const Pending& p) {
    for (auto& x : p.waits) {
        if (!x.value) continue;
        auto it = g_sent.find(x.s);
        if (it != g_sent.end() && it->second >= x.value) continue;
        if (counter(x.s) >= x.value) continue;
        return false;
    }
    return true;
}

void send(Pending& p) {
    HostQueue& h = *p.h;
    std::vector<VkSemaphoreSubmitInfo> w, s;
    for (auto& x : p.waits) {
        VkSemaphoreSubmitInfo i{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        i.semaphore = x.s, i.value = x.value, i.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        w.push_back(i);
    }
    for (auto& x : p.signals) {
        if (x.value && x.s != h.timeline && counter(x.s) >= x.value) continue;
        VkSemaphoreSubmitInfo i{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        i.semaphore = x.s, i.value = x.value, i.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        s.push_back(i);
    }
    VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    si.commandBufferInfoCount = uint32_t(p.ci.size());
    si.pCommandBufferInfos = p.ci.data();
    si.waitSemaphoreInfoCount = uint32_t(w.size());
    si.pWaitSemaphoreInfos = w.data();
    si.signalSemaphoreInfoCount = uint32_t(s.size());
    si.pSignalSemaphoreInfos = s.data();
    VkResult r;
    {
        std::lock_guard q(h.qlock);
        r = vk::vkQueueSubmit2(h.q, 1, &si, VK_NULL_HANDLE);
    }
    if (r == VK_ERROR_DEVICE_LOST) device_lost("submitting work");
    if (r != VK_SUCCESS) {
        static int n = 0;
        if (n++ < 8) KL_ERROR("vk", "queue submission failed: %s", vk::result_text(r));
        VkSemaphoreSignalInfo sg{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        sg.semaphore = h.timeline, sg.value = p.value;
        vk::vkSignalSemaphore(dev(), &sg);
    }
    for (auto& x : p.signals)
        if (x.value) {
            auto& v = g_sent[x.s];
            if (x.value > v) v = x.value;
        }
    p.sent = true;
}

void pump() {
    for (bool again = true; again;) {
        again = false;
        bool blocked[2] = {};
        for (auto it = g_pending.begin(); it != g_pending.end(); ++it) {
            int qi = int((*it)->h - g_q);
            if (blocked[qi]) continue;
            if (!ready(**it)) {
                blocked[qi] = true;
                continue;
            }
            note("q%d submit %llu sent (it was held)", qi, (unsigned long long)(*it)->value);
            send(**it);
            g_pending.erase(it);
            again = true;
            break;
        }
    }
    g_pending_n.store(g_pending.size());
    g_sub_cv.notify_all();
}

std::shared_ptr<Pending> submit(HostQueue& h, const std::vector<Cb*>& cbs, const std::vector<Semaphore>& waits,
                                std::vector<Semaphore> signals) {
    auto p = std::make_shared<Pending>();
    p->h = &h;
    for (Cb* c : cbs) {
        vk::vkEndCommandBuffer(c->cb);
        VkCommandBufferSubmitInfo x{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        x.commandBuffer = c->cb;
        p->ci.push_back(x);
    }
    p->value = h.submitted.load() + 1;
    signals.push_back({h.timeline, p->value});
    p->signals = std::move(signals);
    p->waits = waits;
    if (h.gate_value) p->waits.push_back({h.gate, h.gate_value});
    h.submitted.store(p->value);
    for (Cb* c : cbs) c->value = p->value, h.busy.push_back(c);
    std::lock_guard l(g_sub_lock);
    bool behind = false;
    for (auto& x : g_pending) behind |= x->h == &h;
    bool now = !behind && ready(*p);
    note("q%d submit %llu: %zu command buffers, %zu waits, %zu signals%s", int(&h - g_q), (unsigned long long)p->value, cbs.size(),
         p->waits.size(), p->signals.size(), now ? "" : ", held until its waits are signalled");
    if (now) {
        send(*p);
        if (!g_pending.empty()) pump();
    } else {
        ++g_held;
        g_pending.push_back(p);
        g_pending_n.store(g_pending.size());
    }
    return p;
}

void kick_held() {
    std::lock_guard l(g_sub_lock);
    if (!g_pending.empty()) pump();
}

// More urgent pipeline builds queued than the workers clear in a few seconds.
std::atomic<uint64_t> g_backlog_at{32};
bool backlogged() { return stats::g_counters[stats::CompileUrgent].load(std::memory_order_relaxed) > g_backlog_at.load(std::memory_order_relaxed); }

// Opens each queue's gates in order, once the memory compares as asked (D3D12 comparison
// functions, as VIctory's forks and XCL read them) or a fence is reached; past a bound the gate
// opens anyway: what should write it may have been dropped, and the GPU must not stall for it
// (sooner during a build backlog, when draws and dispatches are being skipped).
void gatekeeper(HostQueue& h) {
    for (;;) {
        Gate g;
        {
            std::unique_lock l(h.gates_lock);
            h.gates_cv.wait(l, [&] { return !h.gates.empty(); });
            g = h.gates.front();
        }
        if (!g.sem) {
            static std::atomic<uint32_t> seen{0};
            if (g.w.compare < 32 && !(seen.fetch_or(1u << g.w.compare) & (1u << g.w.compare)))
                KL_INFO("gpu", "GPU waits on memory with comparison %llu run (polled on the CPU)", (unsigned long long)g.w.compare);
        }
        auto limit = std::chrono::milliseconds(g.sem ? 3000 : backlogged() ? 30 : 250);
        auto t0 = std::chrono::steady_clock::now();
        bool late = false;
        while (!gate_met(g)) {
            auto dt = std::chrono::steady_clock::now() - t0;
            if (dt > limit) {
                late = true;
                h.opened_late++;
                static std::atomic<int> n{0};
                if (n++ < 16) {
                    if (g.sem) {
                        uint64_t v = 0;
                        vk::vkGetSemaphoreCounterValue(dev(), g.sem, &v);
                        KL_WARN("gpu", "a queue wait for fence value %llu (fence at %llu, nothing signals it yet) gave up after 3 s",
                                (unsigned long long)g.sem_value, (unsigned long long)v);
                    } else {
                        bool ok;
                        uint64_t v = read_value(g.w, ok);
                        KL_WARN("gpu", "a GPU wait on memory %llx (comparison %llu, %u bytes, value %llx, wanted %llx) gave up after %lld ms",
                                (unsigned long long)g.w.address, (unsigned long long)g.w.compare, g.w.bytes, (unsigned long long)v,
                                (unsigned long long)g.w.reference, (long long)limit.count());
                    }
                }
                break;
            }
            if (dt < std::chrono::milliseconds(2)) std::this_thread::yield();
            else Sleep(1);
        }
        note("q%d gate %llu opened%s", int(&h - g_q), (unsigned long long)g.opens, late ? " after giving up" : "");
        VkSemaphoreSignalInfo sg{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        sg.semaphore = h.gate, sg.value = g.opens;
        VkResult r = vk::vkSignalSemaphore(dev(), &sg);
        if (r == VK_ERROR_DEVICE_LOST) device_lost("opening a GPU wait");
        {
            std::lock_guard l(h.gates_lock);
            h.gates.pop_front();
        }
        kick_held();
    }
}

void push_gate(HostQueue& h, Gate g) {
    {
        std::lock_guard l(h.gates_lock);
        g.opens = ++h.gate_value;
        g.since = GetTickCount64();
        if (g.sem)
            note("q%d gate %llu: fence %llu", int(&h - g_q), (unsigned long long)g.opens, (unsigned long long)g.sem_value);
        else
            note("q%d gate %llu: memory %llx cmp %llu ref %llx", int(&h - g_q), (unsigned long long)g.opens,
                 (unsigned long long)g.w.address, (unsigned long long)g.w.compare, (unsigned long long)g.w.reference);
        h.gates.push_back(g);
    }
    h.gates_cv.notify_one();
}

void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vk::vkCmdPipelineBarrier2(cb, &di);
}

