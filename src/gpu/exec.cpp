// keylime GPU: title command lists -> Vulkan command buffers (see exec.h).
#include "gpu/exec.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <functional>
#include <condition_variable>
#include <cstdarg>
#include <unordered_map>
#include <vector>

#include "gpu/descriptor.h"
#include "gpu/host.h"
#include "gpu/images.h"
#include "gpu/list.h"
#include "gpu/memory.h"
#include "gpu/misc.h"
#include "gpu/pipeline.h"
#include "gpu/resource.h"
#include "gpu/retire.h"
#include "gpu/root.h"
#include "gpu/stream.h"
#include "gpu/texture.h"
#include "log.h"
#include "shader/container.h"
#include "shader/indirect.h"
#include "shader/rectlist.h"
#include "shader/translate.h"
#include "stats.h"

namespace kl::gpu {
namespace exec {
namespace {

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
};
constexpr uint32_t STAMPS = 1024;
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

// Opens each queue's gates in order, once the memory compares as asked (D3D12 comparison
// functions, as VIctory's forks and XCL read them) or a fence is reached; past a bound the gate
// opens anyway: what should write it may have been dropped, and the GPU must not stall for it.
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
        auto limit = std::chrono::milliseconds(g.sem ? 3000 : 250);
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
                        KL_WARN("gpu", "a GPU wait on memory %llx (comparison %llu, %u bytes, value %llx, wanted %llx) gave up after 250 ms",
                                (unsigned long long)g.w.address, (unsigned long long)g.w.compare, g.w.bytes, (unsigned long long)v,
                                (unsigned long long)g.w.reference);
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

// ------------------------------------------------------------ pipelines ----
struct Compiled {
    uint64_t key = 0;                     // the shaders' hashes: names the pipeline across runs
    std::atomic<bool> used{false};        // drawn or dispatched with this run
    std::atomic<bool> bumped{false};      // queued again ahead of early starts by a first use (once)
    std::atomic<bool> fast_queued{false}; // a dispatch asked for the unoptimized build
    std::atomic<bool> claimed{false};     // a thread has started translating
    std::atomic<bool> translated{false};  // shaders translated (or failed): `ok` and the modules are set
    std::atomic<bool> ready{false};       // nothing more will be built (compute: the optimized pipeline is in, or failed)
    bool ok = false;
    shader::Container vsc, psc, csc;
    const shader::Program *vs = nullptr, *ps = nullptr, *cs = nullptr;
    VkShaderModule vsm = VK_NULL_HANDLE, psm = VK_NULL_HANDLE, csm = VK_NULL_HANDLE;
    std::vector<VkVertexInputAttributeDescription> attrs;
    std::vector<VkVertexInputBindingDescription> binds;
    std::atomic<VkPipeline> compute{VK_NULL_HANDLE};
    std::atomic<VkPipeline> fast{VK_NULL_HANDLE};  // unoptimized, built by a dispatch that could not wait
    std::atomic<bool> fast_claimed{false};
    std::atomic<bool> early{false}, counted{false};
    std::unordered_map<uint64_t, VkPipeline> variants;  // under the state's lock
    std::unordered_set<uint64_t> pending;                // variants a worker is building
    std::unordered_set<uint64_t> urgent_variants;        // variants an urgent job was queued for
    bool writes_color = false, mesh = false;
    uint32_t code_kb = 0, spirv_kb = 0, blocks = 0, guarded = 0;
    std::string plan_note;
    std::vector<uint8_t> vs_rows;                          // VS output locations
    VkShaderModule gs[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};  // rectangle / quad expansion
};

constexpr uint32_t RECTLIST = 17, QUADLIST = 19;

const shader::Program* find(const shader::Container& c, shader::HwStage s) {
    for (auto& p : c.programs)
        if (p.stage == s) return &p;
    return nullptr;
}

bool icase_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (tolower(uint8_t(a[i])) != tolower(uint8_t(b[i]))) return false;
    return true;
}

uint64_t module_key(const shader::Program& p, const shader::Options& o, uint64_t hash) {
    uint64_t h = shader::translator_build() ^ hash;
    auto mix = [&](uint64_t v) { h = (h ^ v) * 0x100000001b3ull; };
    mix(uint64_t(p.stage));
    mix(o.sgpr_offset);
    mix(o.flat ? 0xf1a7 : 0);
    for (auto& [a, b] : o.ps_location) mix(uint64_t(a) << 8 | b);
    return h;
}

std::string spirv_path(uint64_t key) {
    char b[64];
    snprintf(b, sizeof b, "cache\\spirv\\%016llx.bin", (unsigned long long)key);
    return b;
}

constexpr uint32_t SPIRV_MAGIC = 0x3156534b;

bool load_spirv(uint64_t key, shader::Translation& t) {
    FILE* f = fopen(spirv_path(key).c_str(), "rb");
    if (!f) return false;
    bool ok = true;
    auto u32 = [&] {
        uint32_t v = 0;
        ok = ok && fread(&v, 4, 1, f) == 1;
        return v;
    };
    auto bytes = [&](std::vector<uint8_t>& v) {
        uint32_t n = u32();
        if (!ok || n > 4096) return void(ok = false);
        v.resize(n);
        ok = ok && (!n || fread(v.data(), 1, n, f) == n);
    };
    if (u32() != SPIRV_MAGIC) ok = false;
    uint32_t words = u32();
    if (ok && words && words < (64u << 20)) {
        t.spirv.resize(words);
        ok = fread(t.spirv.data(), 4, words, f) == words;
    } else {
        ok = false;
    }
    for (auto& l : t.local_size) l = u32();
    bytes(t.vs_input_rows);
    bytes(t.vs_output_rows);
    bytes(t.ps_input_rows);
    uint32_t flags = u32();
    t.mrt_mask = uint8_t(flags), t.writes_depth = flags & 0x100, t.discards = flags & 0x200;
    t.blocks = u32();
    t.guarded_blocks = u32();
    fclose(f);
    return ok && t.spirv[0] == 0x07230203;
}

void save_spirv(uint64_t key, const shader::Translation& t) {
    CreateDirectoryA("cache", nullptr);
    CreateDirectoryA("cache\\spirv", nullptr);
    std::string path = spirv_path(key), tmp = path + "." + std::to_string(GetCurrentThreadId()) + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) return;
    auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    auto bytes = [&](const std::vector<uint8_t>& v) {
        u32(uint32_t(v.size()));
        if (!v.empty()) fwrite(v.data(), 1, v.size(), f);
    };
    u32(SPIRV_MAGIC);
    u32(uint32_t(t.spirv.size()));
    fwrite(t.spirv.data(), 4, t.spirv.size(), f);
    for (auto l : t.local_size) u32(l);
    bytes(t.vs_input_rows);
    bytes(t.vs_output_rows);
    bytes(t.ps_input_rows);
    u32(t.mrt_mask | (t.writes_depth ? 0x100 : 0) | (t.discards ? 0x200 : 0));
    u32(t.blocks);
    u32(t.guarded_blocks);
    bool ok = fclose(f) == 0;
    if (ok) MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
    else DeleteFileA(tmp.c_str());
}

VkShaderModule make_module(const shader::Translation& t) {
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = t.spirv.size() * 4;
    mi.pCode = t.spirv.data();
    VkShaderModule m = VK_NULL_HANDLE;
    KL_VK_CHECK(vk::vkCreateShaderModule(dev(), &mi, nullptr, &m));
    return m;
}

VkShaderModule module(const shader::Program& p, const shader::Options& given, uint64_t hash, shader::Translation& t) {
    std::string err;
    shader::Options o = given;
    o.flat = memory::flat();
    uint64_t key = module_key(p, o, hash);
    memory::name_shader(shader::code_id(p), hash, p.stage == shader::HwStage::Ps ? 1 : p.stage == shader::HwStage::Cs ? 2 : 0);
    if (load_spirv(key, t)) {
        stats::add(stats::ShadersCached);
        return make_module(t);
    }
    t = shader::Translation{};
    auto t0 = std::chrono::steady_clock::now();
    bool translated = shader::translate(p, t, err, &o);
    stats::add(stats::TranslateUs, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()));
    stats::add(stats::ShadersTranslated);
    stats::add(stats::SpirvBytes, t.spirv.size() * 4);
    stats::add(stats::BlocksTotal, t.blocks);
    stats::add(stats::BlocksGuarded, t.guarded_blocks);
    if (!translated) {
        static int n = 0;
        if (n++ < 32) KL_WARN("gpu", "shader %016llx (%s) not translated: %s", (unsigned long long)hash, shader::stage_name(p.stage), err.c_str());
        return VK_NULL_HANDLE;
    }
    save_spirv(key, t);
    return make_module(t);
}

// Pipelines are built by workers: one driver compile can take many seconds, and the queue
// thread (the title's) never makes one. A first use queues the pipeline ahead of the rest and
// waits for it within a small budget per frame over all threads (4 ms for draws); past it, draws
// whose pipeline is not ready are dropped. Dispatches always wait: later work, the title's GPU
// waits and its CPU read results they write, and garbage there crashes the game. A compute
// pipeline gets a quick unoptimized build first (well under a second), then the optimized one. Pipelines an earlier run used (listed in
// cache\used-pipelines.txt) start building as soon as the title creates them.
std::mutex g_jobs_lock, g_done_lock;
std::condition_variable g_jobs_cv, g_done_cv;
// g_urgent: what a draw or dispatch waits for now (first-use translations, quick compute
// builds). Some workers take only those, so a backlog of optimized builds (30-90 s each for
// big translated compute shaders) can never hold up a dispatch the title is waiting on.
// g_warm: early starts (translation and quick build of pipelines earlier runs used), taken by
// every worker whenever nothing urgent waits, so they are ready before the scene needs them;
// g_jobs: optimized builds, on two low-priority workers during play.
std::deque<std::function<void()>> g_jobs, g_urgent, g_warm;
void notify() {
    { std::lock_guard g(g_done_lock); }
    g_done_cv.notify_all();
}

// Driver pipeline cache, kept in cache\pipelines.bin across runs: written every 30 s while
// new pipelines are being built.
constexpr const char* CACHE_FILE = "cache\\pipelines.bin";
void open_cache() {
    std::vector<uint8_t> data;
    if (FILE* f = fopen(CACHE_FILE, "rb")) {
        _fseeki64(f, 0, SEEK_END);  // the cache can pass 2 GB: 64-bit offsets
        long long n = _ftelli64(f);
        _fseeki64(f, 0, SEEK_SET);
        if (n > 0) data.resize(size_t(n));
        if (!data.empty() && fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
        fclose(f);
    }
    VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    ci.initialDataSize = data.size();
    ci.pInitialData = data.empty() ? nullptr : data.data();
    if (vk::vkCreatePipelineCache(dev(), &ci, nullptr, &g_cache) != VK_SUCCESS) {
        ci.initialDataSize = 0, ci.pInitialData = nullptr;
        KL_VK_CHECK(vk::vkCreatePipelineCache(dev(), &ci, nullptr, &g_cache));
    }
    KL_INFO("gpu", "pipeline cache: %zu KB loaded from %s", data.size() >> 10, CACHE_FILE);
}
void save_cache(bool force = false) {
    static std::mutex m;
    static auto last = std::chrono::steady_clock::now();
    static uint64_t saved_at = 0;
    std::unique_lock g(m, std::try_to_lock);
    if (!g || !g_cache) return;
    uint64_t built = stats::g_counters[stats::PipelinesBuilt].load();
    auto now = std::chrono::steady_clock::now();
    if (built == saved_at || (!force && now - last < std::chrono::seconds(30))) return;
    last = now, saved_at = built;
    size_t n = 0;
    static std::atomic<int> fails{0};
    auto fail = [&](const char* what, long long detail) {
        if (fails++ < 4) KL_WARN("gpu", "pipeline cache not saved: %s (%lld)", what, detail);
    };
    // Workers keep adding pipelines while this copies: room to spare, and again if it grew past it
    // (VK_INCOMPLETE, which every save of the 3 GB cache hit before).
    std::vector<uint8_t> data;
    VkResult r = VK_INCOMPLETE;
    for (int tries = 0; tries < 4 && r == VK_INCOMPLETE; ++tries) {
        r = vk::vkGetPipelineCacheData(dev(), g_cache, &n, nullptr);
        if (r != VK_SUCCESS || !n) return fail("size query", r);
        n += n / 16 + (64 << 20);
        data.resize(n);
        r = vk::vkGetPipelineCacheData(dev(), g_cache, &n, data.data());
    }
    if (r != VK_SUCCESS) return fail("data", r);
    CreateDirectoryA("cache", nullptr);
    FILE* f = fopen("cache\\pipelines.tmp", "wb");
    if (!f) return fail("open cache\\pipelines.tmp", errno);
    bool ok = fwrite(data.data(), 1, n, f) == n;
    fclose(f);
    if (!ok) return fail("write", (long long)n);
    if (!MoveFileExA("cache\\pipelines.tmp", CACHE_FILE, MOVEFILE_REPLACE_EXISTING)) return fail("replace", GetLastError());
    static std::atomic<int> told{0};
    if (told++ < 3) KL_INFO("gpu", "pipeline cache saved: %zu KB", n >> 10);
}
void save_used(bool force = false);
std::atomic<bool> g_precompiling{false};  // `keylime precompile` is running
// `front`: ahead of other urgent work (graphics variants build in milliseconds; quick compute builds
// can take seconds on a cold cache).
void job(std::function<void()> f, bool urgent = false, bool front = false, bool warm = false) {
    stats::add(stats::Compiling);
    static std::once_flag once;
    std::call_once(once, [] {
        // Background (optimized) builds take up to a minute of CPU each; during play they get 2
        // low-priority workers so the quick builds a dispatch waits for keep the CPU (with 16
        // of them, quick builds took 35-60 s in the opening cutscene). `keylime precompile`
        // uses every core.
        unsigned cores = std::max(4u, std::thread::hardware_concurrency());
        unsigned n = g_precompiling ? std::min(16u, cores * 2 / 3) : 2;
        const unsigned URGENT_ONLY = g_precompiling ? 8 : std::max(4u, std::min(16u, cores / 2));
        // Kinds of worker: 4 strictly urgent (never busy with a long build when a dispatch needs
        // one), the rest of the urgent pool urgent then warm, and n background workers urgent,
        // warm, then optimized builds.
        constexpr unsigned STRICT_URGENT = 4;
        for (unsigned i = 0; i < n + URGENT_ONLY; ++i)
            std::thread([urgent_only = i >= n, strict = i >= n + URGENT_ONLY - std::min(STRICT_URGENT, URGENT_ONLY)] {
                SetThreadDescription(GetCurrentThread(), strict ? L"keylime-compile-strict" : urgent_only ? L"keylime-compile-urgent" : L"keylime-compile");
                SetThreadPriority(GetCurrentThread(), strict ? THREAD_PRIORITY_ABOVE_NORMAL : urgent_only ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_LOWEST);
                for (;;) {
                    std::function<void()> f;
                    {
                        std::unique_lock g(g_jobs_lock);
                        g_jobs_cv.wait(g, [urgent_only, strict] {
                            return !g_urgent.empty() || (!strict && !g_warm.empty()) || (!urgent_only && !g_jobs.empty());
                        });
                        auto& from = !g_urgent.empty() ? g_urgent : (!strict && !g_warm.empty()) ? g_warm : g_jobs;
                        f = std::move(from.front());
                        from.pop_front();
                        stats::set(stats::CompileQueue, g_jobs.size() + g_urgent.size() + g_warm.size());
                    }
                    f();
                    stats::g_counters[stats::Compiling].fetch_sub(1);
                    notify();
                    save_cache();
                    save_used();
                }
            }).detach();
    });
    {
        std::lock_guard g(g_jobs_lock);
        if (urgent && front) g_urgent.push_front(std::move(f));
        else if (urgent) g_urgent.push_back(std::move(f));
        else if (warm) g_warm.push_back(std::move(f));
        else g_jobs.push_back(std::move(f));
        stats::set(stats::CompileQueue, g_jobs.size() + g_urgent.size() + g_warm.size());
    }
    // notify_one could wake an urgent-only worker for normal work and lose the wakeup.
    g_jobs_cv.notify_all();
}
// Waiting allowed per shown frame, over all threads: [0] draws, [1] dispatches. Past it, a
// draw or dispatch whose pipeline is still compiling is skipped this frame (as VIctory does with
// its shared 500 ms budget) instead of holding the title until hundreds of compiles finish.
// KEYLIME_DISPATCH_WAIT_MS sets the dispatch budget (default 500).
const uint64_t BUDGET_US[2] = {4000, [] {
                                   char v[16]{};
                                   return uint64_t(GetEnvironmentVariableA("KEYLIME_DISPATCH_WAIT_MS", v, 16) ? atoi(v) : 500) * 1000;
                               }()};
std::atomic<uint64_t> g_budget_frame[2]{~0ull, ~0ull}, g_waited_us[2]{}, g_waited_total{0};
template <class F>
bool wait_budget(F done, int kind) {
    if (done()) return true;
    uint64_t frame = stats::g_counters[stats::FramesShown].load(std::memory_order_relaxed);
    if (g_budget_frame[kind].exchange(frame) != frame) g_waited_us[kind].store(0);
    uint64_t waited = g_waited_us[kind].load();
    if (waited >= BUDGET_US[kind]) return false;
    auto t0 = std::chrono::steady_clock::now();
    bool ok;
    {
        std::unique_lock g(g_done_lock);
        ok = g_done_cv.wait_until(g, t0 + std::chrono::microseconds(BUDGET_US[kind] - waited), done);
    }
    uint64_t us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
    g_waited_us[kind] += us;
    g_waited_total += us;
    stats::add(stats::CompileWaitUs, us);
    return ok;
}
void dropped(const char* what) {
    static std::atomic<bool> told{false};
    if (!told.exchange(true))
        KL_INFO("gpu", "draws whose pipeline is not ready within 4 ms a frame, and dispatches past %llu ms a frame, are skipped meanwhile",
                (unsigned long long)(BUDGET_US[1] / 1000));
    stats::add(stats::WaitingOnCompile);
    if (what[1] != 'r') stats::add(stats::DispatchesSkipped);
}

// Pipelines used in earlier runs, and those used in this one.
constexpr const char* USED_FILE = "cache\\used-pipelines.txt";
std::mutex g_used_lock;
std::unordered_set<uint64_t> g_used_before, g_used_now;
std::atomic<uint64_t> g_used_added{0}, g_started_early{0};
std::chrono::steady_clock::time_point g_run_start = std::chrono::steady_clock::now();
void precompiled(Compiled* k) {
    if (!k->early.load() || k->counted.exchange(true)) return;
    stats::add(stats::PrecompileDone);
    uint64_t done = stats::g_counters[stats::PrecompileDone].load(), listed = stats::g_counters[stats::PrecompileListed].load();
    if (done == listed || done % 1000 == 0)
        KL_INFO("gpu", "precompile: %llu of the %llu pipelines used in earlier runs are built, %.0f s after start", (unsigned long long)done,
                (unsigned long long)listed, std::chrono::duration<double>(std::chrono::steady_clock::now() - g_run_start).count());
}
uint64_t pipeline_key(const pipeline::State* st) {
    uint64_t h = st->compute ? 0x9e3779b97f4a7c15ull : 0xcbf29ce484222325ull;
    for (auto& s : st->shaders) h = (h ^ s.hash ^ (uint64_t(s.stage) << 56)) * 0x100000001b3ull;
    return h;
}
void open_used() {
    if (FILE* f = fopen(USED_FILE, "r")) {
        unsigned long long k;
        while (fscanf(f, "%llx", &k) == 1) g_used_before.insert(k);
        fclose(f);
    }
    KL_INFO("gpu", "pipelines: %zu used in earlier runs (%s) start as soon as the game creates them", g_used_before.size(), USED_FILE);
    stats::set(stats::PrecompileListed, g_used_before.size());
}
void note_used(Compiled* k) {
    if (k->used.exchange(true)) return;
    std::lock_guard g(g_used_lock);
    if (!g_used_before.count(k->key) && g_used_now.insert(k->key).second) ++g_used_added;
}
void save_used(bool force) {
    static std::mutex m;
    static auto last = std::chrono::steady_clock::now();
    static uint64_t saved_at = 0;
    std::unique_lock g(m, std::try_to_lock);
    if (!g) return;
    uint64_t added = g_used_added.load();
    auto now = std::chrono::steady_clock::now();
    if (added == saved_at || (!force && now - last < std::chrono::seconds(1))) return;
    last = now, saved_at = added;
    std::vector<uint64_t> keys;
    {
        std::lock_guard u(g_used_lock);
        keys.assign(g_used_before.begin(), g_used_before.end());
        keys.insert(keys.end(), g_used_now.begin(), g_used_now.end());
    }
    CreateDirectoryA("cache", nullptr);
    if (FILE* f = fopen("cache\\used-pipelines.tmp", "w")) {
        for (uint64_t k : keys) fprintf(f, "%016llx\n", (unsigned long long)k);
        bool ok = fclose(f) == 0;
        if (ok) MoveFileExA("cache\\used-pipelines.tmp", USED_FILE, MOVEFILE_REPLACE_EXISTING);
    }
    static uint64_t logged = 0;
    if (force || added - logged >= 16) {
        logged = added;
        KL_INFO("gpu", "pipelines: %zu listed as used (%llu new this run, %llu started early)", keys.size(), (unsigned long long)added,
                (unsigned long long)g_started_early.load());
    }
}

void shaders(pipeline::State* st, Compiled* k);
VkPipeline compute_pipeline(Compiled* k, bool fast, bool only_cached = false);
void prepare_and_submit(HostQueue& h, int qi, std::vector<Cb*> cbs);
void optimize(Compiled* k);
bool prebuild(pipeline::State* st, Compiled* k);

// A worker translates the shaders; a compute pipeline a dispatch waits for gets a quick
// unoptimized build first, then the optimized one (queued behind what draws wait for).
void fast_build(Compiled* k);
VkPipeline compute_pipeline(Compiled* k, bool fast, bool only_cached);
// The optimized pipeline if the driver's cache already holds it (no compile).
bool cached_optimized(Compiled* k) {
    if (!host::get().caps.cache_control) return false;
    VkPipeline p = compute_pipeline(k, false, true);
    if (!p) return false;
    k->compute.store(p, std::memory_order_release);
    k->ready.store(true, std::memory_order_release);
    stats::add(stats::PipelinesBuilt);
    precompiled(k);
    notify();
    return true;
}
void translate_claimed(pipeline::State* st, Compiled* k, bool first_use) {
    shaders(st, k);
    k->translated.store(true, std::memory_order_release);
    if (!st->compute || !k->ok) k->ready.store(true, std::memory_order_release);
    if (!k->ok) precompiled(k);
    notify();
    if (st->compute && k->ok) {
        // Already in the driver's cache (an earlier run or `keylime precompile`): ready now.
        if (cached_optimized(k)) return;
        // Early starts too: a quick build makes it runnable within about a second, the
        // optimized one (up to a minute for big translated shaders) follows on a worker.
        fast_build(k);
        if (g_precompiling) {
            // `keylime precompile`: the quick build is what a run needs at once (about a second
            // each on all cores); optimized builds follow during play.
            k->ready.store(true, std::memory_order_release);
            notify();
            return;
        }
        job([k] { optimize(k); });
    } else if (k->ok) {
        if (!prebuild(st, k)) precompiled(k);
    }
}

void queue_translation(pipeline::State* st, Compiled* k, bool first_use) {
    job([st, k, first_use] {
        if (!k->claimed.exchange(true)) translate_claimed(st, k, first_use);
    }, first_use, first_use && !st->compute, !first_use);  // a draw's shaders translate in milliseconds: ahead of quick compute builds
}

// `first_use`: a draw or dispatch wants it now (ahead of early starts).
Compiled* obtain(pipeline::State* st, bool first_use) {
    Compiled* k;
    bool fresh = false;
    {
        std::lock_guard g(st->lock);
        k = static_cast<Compiled*>(st->host);
        if (!k) {
            k = new Compiled();
            k->key = pipeline_key(st);
            st->host = k;
            fresh = true;
        }
    }
    if (fresh) queue_translation(st, k, first_use);
    else if (first_use && !k->claimed.load() && !k->bumped.exchange(true)) queue_translation(st, k, true);
    if (first_use) note_used(k);
    return k;
}

void optimize(Compiled* k) {
    auto t0 = std::chrono::steady_clock::now();
    VkPipeline p = compute_pipeline(k, false);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec > 2)
        KL_WARN("gpu", "compute pipeline %016llx took %.1f s to compile (%u KB of console code, %u KB SPIR-V, %u blocks, %u run by number%s%s)",
                (unsigned long long)k->key, sec, k->code_kb, k->spirv_kb, k->blocks, k->guarded, k->plan_note.empty() ? "" : ": ",
                k->plan_note.c_str());
    stats::add(stats::PipelinesBuilt);
    stats::add(stats::OptimizedBuilt);
    stats::add(stats::OptimizedUs, uint64_t(sec * 1e6));
    stats::raise(stats::OptimizedWorstUs, uint64_t(sec * 1e6));
    if (p) {
        k->compute.store(p, std::memory_order_release);
        if (VkPipeline f = k->fast.exchange(VK_NULL_HANDLE)) retire([f] { vk::vkDestroyPipeline(dev(), f, nullptr); });
    }
    k->ready.store(true, std::memory_order_release);
    precompiled(k);
    notify();
}

void fast_build(Compiled* k) {
    if (k->fast_claimed.exchange(true) || k->compute.load(std::memory_order_acquire)) return;
    auto t0 = std::chrono::steady_clock::now();
    VkPipeline p = compute_pipeline(k, true);
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sec > 0.5) KL_WARN("gpu", "an unoptimized compute pipeline took %.2f s to compile", sec);
    stats::add(stats::QuickBuilt);
    stats::add(stats::QuickUs, uint64_t(sec * 1e6));
    if (!k->compute.load(std::memory_order_acquire)) k->fast.store(p, std::memory_order_release);
    else if (p) retire([p] { vk::vkDestroyPipeline(dev(), p, nullptr); });
    notify();
}

// A compute pipeline to run now, within the frame's wait budget: the optimized one, else the
// unoptimized one. `pending`: not ready yet (rather than impossible to build).
VkPipeline runnable(pipeline::State* st, bool& pending) {
    Compiled* k = obtain(st, true);
    pending = false;
    if (k->translated.load(std::memory_order_acquire) && k->ok && !k->compute.load() && !k->fast_claimed.load() &&
        !k->fast_queued.exchange(true))
        job([k] { fast_build(k); }, true);
    auto have = [k] {
        if (!k->translated.load(std::memory_order_acquire)) return false;
        return !k->ok || k->compute.load() || k->fast.load() || k->ready.load();
    };
    if (!wait_budget(have, 1)) {
        pending = true;
        return VK_NULL_HANDLE;
    }
    if (!k->ok) return VK_NULL_HANDLE;
    VkPipeline p = k->compute.load(std::memory_order_acquire);
    return p ? p : k->fast.load(std::memory_order_acquire);
}

VkPipeline compute_pipeline(Compiled* k, bool fast, bool only_cached) {
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.flags = fast ? VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT : 0;
    if (only_cached) ci.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, k->csm, "main", nullptr};
    ci.layout = g_layout;
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = vk::vkCreateComputePipelines(dev(), g_cache, 1, &ci, nullptr, &p);
    if (only_cached && r != VK_SUCCESS) return VK_NULL_HANDLE;
    if (r != VK_SUCCESS) {
        static int n = 0;
        if (n++ < 16) KL_WARN("gpu", "compute pipeline not built by the driver%s: %s", fast ? " (unoptimized)" : "", vk::result_text(r));
        return VK_NULL_HANDLE;
    }
    return p;
}

void shaders(pipeline::State* st, Compiled* k) {
    const pipeline::Shader *vs = nullptr, *ps = nullptr, *cs = nullptr, *as = nullptr, *ms = nullptr;
    for (auto& s : st->shaders) {
        if (s.stage == 0) vs = &s;
        if (s.stage == 1) ps = &s;
        if (s.stage == 5) cs = &s;
        if (s.stage == 6) as = &s;
        if (s.stage == 7) ms = &s;
    }
    std::string err;
    if (st->compute) {
        if (!cs || !shader::parse(cs->code.data(), cs->code.size(), k->csc, err) || !(k->cs = find(k->csc, shader::HwStage::Cs))) {
            KL_WARN("gpu", "compute pipeline without a usable CS: %s", err.c_str());
            return;
        }
        shader::Translation t;
        k->csm = module(*k->cs, {}, cs->hash, t);
        k->ok = k->csm != VK_NULL_HANDLE;
        k->code_kb = uint32_t(k->cs->code.size() >> 10), k->spirv_kb = uint32_t((t.spirv.size() * 4) >> 10);
        k->blocks = t.blocks, k->guarded = t.guarded_blocks, k->plan_note = t.plan_note;
        return;
    }
    if (as) {
        static int n = 0;
        if (n++ < 4) KL_WARN("gpu", "mesh pipelines with an amplification shader are skipped (not supported yet)");
        return;
    }
    if (ms && !host::get().caps.mesh_shader) {
        static int n = 0;
        if (n++ < 4) KL_WARN("gpu", "the driver has no VK_EXT_mesh_shader, so mesh pipelines are skipped");
        return;
    }
    k->mesh = ms != nullptr;
    if (ms) vs = ms;
    shader::HwStage front = k->mesh ? shader::HwStage::Ms : shader::HwStage::Vs;
    if (!vs || !shader::parse(vs->code.data(), vs->code.size(), k->vsc, err) || !(k->vs = find(k->vsc, front))) {
        static int n = 0;
        if (n++ < 16) KL_WARN("gpu", "graphics pipeline without a usable %s (%s)", k->mesh ? "MS" : "VS", err.empty() ? "no program" : err.c_str());
        return;
    }
    shader::Translation vt;
    shader::Options vo;
    vo.sgpr_offset = 0;
    k->vsm = module(*k->vs, vo, vs->hash, vt);
    if (!k->vsm) return;
    k->vs_rows = vt.vs_output_rows;
    if (ps && shader::parse(ps->code.data(), ps->code.size(), k->psc, err) && (k->ps = find(k->psc, shader::HwStage::Ps))) {
        // Link PS inputs to VS outputs by semantic (rows differ when the signatures do).
        shader::Options po;
        po.sgpr_offset = 32;
        for (auto& in : k->psc.inputs) {
            if (in.system_value) continue;
            for (auto& out : k->vsc.outputs)
                if (!out.system_value && out.index == in.index && icase_equal(out.name, in.name)) {
                    po.ps_location[uint8_t(in.reg)] = uint8_t(out.reg);
                    break;
                }
        }
        shader::Translation pt;
        k->psm = module(*k->ps, po, ps->hash, pt);
        k->writes_color = pt.mrt_mask != 0;
        if (!k->psm) return;
    }
    // Vertex input: each VS input row from the input layout element with its semantic.
    uint32_t slots_used = 0;
    for (auto& sig : k->vsc.inputs) {
        if (sig.system_value || k->mesh) continue;
        for (auto& e : st->input) {
            if (e.index != sig.index || !icase_equal(e.semantic, sig.name)) continue;
            VkFormat f = images::vk_format(e.format);
            if (f == VK_FORMAT_UNDEFINED) break;
            k->attrs.push_back({sig.reg, e.slot & 31, f, e.offset});
            if (!(slots_used & (1u << (e.slot & 31)))) {
                slots_used |= 1u << (e.slot & 31);
                k->binds.push_back({e.slot & 31, 0, e.per_instance ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX});
            }
            break;
        }
    }
    k->ok = true;
    return;
}

VkBlendFactor blend_factor(uint32_t b) {
    static const VkBlendFactor m[] = {VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_SRC_COLOR,
                                      VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                                      VK_BLEND_FACTOR_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_DST_COLOR,
                                      VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_ZERO,
                                      VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                                      VK_BLEND_FACTOR_SRC1_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR, VK_BLEND_FACTOR_SRC1_ALPHA,
                                      VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA, VK_BLEND_FACTOR_CONSTANT_ALPHA,
                                      VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA};
    return b < sizeof m / sizeof m[0] ? m[b] : VK_BLEND_FACTOR_ONE;
}
VkBlendOp blend_op(uint32_t o) {
    switch (o) {
        case 2: return VK_BLEND_OP_SUBTRACT;
        case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
        case 4: return VK_BLEND_OP_MIN;
        case 5: return VK_BLEND_OP_MAX;
        default: return VK_BLEND_OP_ADD;
    }
}
VkCompareOp compare(uint32_t f) { return f >= 1 && f <= 8 ? VkCompareOp(f - 1) : VK_COMPARE_OP_ALWAYS; }
VkStencilOp stencil_op(uint32_t o) { return o >= 1 && o <= 8 ? VkStencilOp(o - 1) : VK_STENCIL_OP_KEEP; }
VkStencilOpState stencil(const pipeline::StencilFace& f, const pipeline::DepthStencil& d) {
    return {stencil_op(f.fail), stencil_op(f.pass), stencil_op(f.depth_fail), compare(f.func), d.read_mask, d.write_mask, 0};
}

// What a draw renders into (the variant key's parts).
struct Attachments {
    uint32_t count = 0;
    VkFormat color[8]{};
    VkFormat depth = VK_FORMAT_UNDEFINED;
    bool stencil = false;
};

uint32_t topology_class(uint32_t t) {  // 0 point 1 line 2 triangle 3 patch, ~0 unknown
    switch (t) {
        case 1: return 0;
        case 2: case 3: case 10: case 11: return 1;
        case 4: case 5: case 12: case 13: case RECTLIST: case QUADLIST: return 2;
        default: return t >= 33 && t <= 64 ? 3 : ~0u;
    }
}
VkPrimitiveTopology vk_topology(uint32_t t) {
    switch (t) {
        case 1: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        case 2: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        case 3: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        case 5: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
        case 10: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
        case 11: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
        case 12: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
        case 13: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
        case QUADLIST: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;  // 4 vertices a primitive, into the GS
        default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

void build_variant(Compiled* k, pipeline::State* st, const Attachments& a, uint32_t topo, uint64_t key);
uint64_t request_variant(Compiled* k, pipeline::State* st, const Attachments& a, uint32_t topo, bool urgent) {
    uint64_t key = 0xcbf29ce484222325ull;
    auto mix = [&](uint64_t v) { key = (key ^ v) * 0x100000001b3ull; };
    int expand = topo == RECTLIST ? 0 : topo == QUADLIST ? 1 : -1;
    mix(a.count), mix(a.depth), mix(a.stencil), mix(topology_class(topo)), mix(uint64_t(expand + 7));
    for (uint32_t i = 0; i < a.count; ++i) mix(a.color[i]);
    {
        std::lock_guard g(st->lock);
        auto it = k->variants.find(key);
        if (it == k->variants.end()) {
            k->variants[key] = VK_NULL_HANDLE;
            k->pending.insert(key);
            if (urgent) k->urgent_variants.insert(key);
            job([k, st, a, topo, key] { build_variant(k, st, a, topo, key); }, urgent, urgent, !urgent);  // early ones: warm tier
        } else if (urgent && k->pending.count(key) && k->urgent_variants.insert(key).second) {
            // An early start asked for it at low priority (behind thousands of background builds): a
            // draw waits now, so it gets an urgent build of its own; the first one done is kept.
            job([k, st, a, topo, key] { build_variant(k, st, a, topo, key); }, true, true);
        }
    }
    return key;
}
VkPipeline variant(Compiled* k, pipeline::State* st, const Attachments& a, uint32_t topo) {
    uint64_t key = request_variant(k, st, a, topo, true);
    VkPipeline out = VK_NULL_HANDLE;
    wait_budget([&] {
        std::lock_guard g(st->lock);
        if (k->pending.count(key)) return false;
        out = k->variants[key];
        return true;
    }, 0);
    return out;
}

void build_variant(Compiled* k, pipeline::State* st, const Attachments& a, uint32_t topo, uint64_t key) {
    {
        std::lock_guard g(st->lock);
        if (!k->pending.count(key)) return;  // the other (urgent or early) build got there first
    }
    int expand = topo == RECTLIST ? 0 : topo == QUADLIST ? 1 : -1;
    auto finish = [&](VkPipeline p) {
        {
            std::lock_guard g(st->lock);
            if (!k->pending.count(key)) {  // built twice: keep the first
                if (p) retire([p] { vk::vkDestroyPipeline(dev(), p, nullptr); });
                return;
            }
            k->variants[key] = p;
            k->pending.erase(key);
        }
        precompiled(k);
    };
    VkPipelineShaderStageCreateInfo stages[3];
    uint32_t nstages = 0;
    stages[nstages++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                         k->mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_VERTEX_BIT, k->vsm, "main", nullptr};
    if (k->mesh) expand = -1;
    if (expand >= 0) {
        std::lock_guard g(st->lock);
        if (!k->gs[expand]) {
            auto code = shader::rect_geometry(k->vs_rows, expand == 1);
            VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            mi.codeSize = code.size() * 4;
            mi.pCode = code.data();
            KL_VK_CHECK(vk::vkCreateShaderModule(dev(), &mi, nullptr, &k->gs[expand]));
        }
        if (!k->gs[expand]) return finish(VK_NULL_HANDLE);
        stages[nstages++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_GEOMETRY_BIT, k->gs[expand], "main", nullptr};
    }
    if (k->psm)
        stages[nstages++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, k->psm, "main", nullptr};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = uint32_t(k->binds.size());
    vi.pVertexBindingDescriptions = k->binds.data();
    vi.vertexAttributeDescriptionCount = uint32_t(k->attrs.size());
    vi.pVertexAttributeDescriptions = k->attrs.data();
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = vk_topology(topo);
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = 1, vp.scissorCount = 1;
    const auto& r = st->raster;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.depthClampEnable = r.depth_clip ? VK_FALSE : VK_TRUE;
    rs.polygonMode = r.fill == 2 ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rs.cullMode = r.cull == 2 ? VK_CULL_MODE_FRONT_BIT : r.cull == 3 ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    rs.frontFace = r.front_ccw ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
    rs.depthBiasEnable = (r.depth_bias || r.slope_bias != 0.0f) ? VK_TRUE : VK_FALSE;
    rs.depthBiasConstantFactor = float(r.depth_bias);
    rs.depthBiasClamp = r.bias_clamp;
    rs.depthBiasSlopeFactor = r.slope_bias;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    ms.alphaToCoverageEnable = st->alpha_to_coverage ? VK_TRUE : VK_FALSE;
    const auto& d = st->depth;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    if (a.depth != VK_FORMAT_UNDEFINED) {
        ds.depthTestEnable = d.depth_enable ? VK_TRUE : VK_FALSE;
        ds.depthWriteEnable = d.depth_enable && d.depth_write ? VK_TRUE : VK_FALSE;
        ds.depthCompareOp = compare(d.depth_func);
        ds.stencilTestEnable = d.stencil_enable && a.stencil ? VK_TRUE : VK_FALSE;
        ds.front = stencil(d.front, d);
        ds.back = stencil(d.back, d);
    }
    VkPipelineColorBlendAttachmentState cb[8]{};
    for (uint32_t i = 0; i < a.count; ++i) {
        const auto& b = st->blend[i];
        cb[i].blendEnable = b.enable ? VK_TRUE : VK_FALSE;
        cb[i].srcColorBlendFactor = blend_factor(b.src);
        cb[i].dstColorBlendFactor = blend_factor(b.dst);
        cb[i].colorBlendOp = blend_op(b.op);
        cb[i].srcAlphaBlendFactor = blend_factor(b.src_alpha);
        cb[i].dstAlphaBlendFactor = blend_factor(b.dst_alpha);
        cb[i].alphaBlendOp = blend_op(b.op_alpha);
        cb[i].colorWriteMask = b.write_mask & 0xf;
    }
    VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    bs.attachmentCount = a.count;
    bs.pAttachments = cb;
    VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_BLEND_CONSTANTS,
                            VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
                            VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE, VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = k->mesh ? 4 : sizeof dyn / sizeof dyn[0];
    dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    ri.colorAttachmentCount = a.count;
    ri.pColorAttachmentFormats = a.color;
    ri.depthAttachmentFormat = a.depth;
    ri.stencilAttachmentFormat = a.stencil ? a.depth : VK_FORMAT_UNDEFINED;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.pNext = &ri;
    ci.stageCount = nstages;
    ci.pStages = stages;
    ci.pVertexInputState = k->mesh ? nullptr : &vi;
    ci.pInputAssemblyState = k->mesh ? nullptr : &ia;
    ci.pViewportState = &vp;
    ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds;
    ci.pColorBlendState = &bs;
    ci.pDynamicState = &dy;
    ci.layout = g_layout;
    VkPipeline p = VK_NULL_HANDLE;
    auto t0 = std::chrono::steady_clock::now();
    VkResult res = vk::vkCreateGraphicsPipelines(dev(), g_cache, 1, &ci, nullptr, &p);
    stats::add(stats::GraphicsBuilt);
    stats::add(stats::GraphicsUs, uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()));
    if (res != VK_SUCCESS) {
        static int n = 0;
        if (n++ < 16) KL_WARN("gpu", "graphics pipeline not built by the driver: %s", vk::result_text(res));
        p = VK_NULL_HANDLE;
    }
    stats::add(stats::PipelinesBuilt);
    finish(p);
}

// The variant the pipeline declares (its render target and depth formats, a list of its
// topology type), built ahead so the first draw likely finds it ready.
bool prebuild(pipeline::State* st, Compiled* k) {
    static const uint32_t topo_of[] = {0, 1, 2, 4};
    if (st->topology_type == 0 || st->topology_type > 3) return false;
    Attachments a;
    a.count = std::min(st->rt_count, 8u);
    for (uint32_t i = 0; i < a.count; ++i) a.color[i] = images::vk_format(st->rt_formats[i]);
    while (a.count && a.color[a.count - 1] == VK_FORMAT_UNDEFINED) --a.count;
    if (st->dsv_format) {
        a.depth = images::vk_format(st->dsv_format);
        a.stencil = a.depth == VK_FORMAT_D16_UNORM_S8_UINT || a.depth == VK_FORMAT_D24_UNORM_S8_UINT || a.depth == VK_FORMAT_D32_SFLOAT_S8_UINT;
    }
    request_variant(k, st, a, k->mesh ? 4 : topo_of[st->topology_type], false);
    return true;
}

// --------------------------------------------------------------- builder ----
struct Arg {
    uint64_t value = 0;
    uint32_t c[64] = {};
};
struct Bind {
    void* root = nullptr;
    pipeline::State* pso = nullptr;
    Arg args[64];
};

template <class... A>
void once(const char* what, A... a) {
    static std::mutex m;
    static std::unordered_map<std::string, int> seen;
    std::lock_guard g(m);
    if (seen[what]++ == 0) KL_WARN("gpu", what, a...);
}

struct Builder {
    HostQueue& hq;
    Cb* c;
    VkCommandBuffer cb;
    Bind bind[2];  // 0 compute, 1 graphics
    uint64_t view_heap = 0, sampler_heap = 0;
    cmd::IndexBuffer ib{};
    cmd::VertexView vb[32]{};
    uint32_t topology = 4;
    cmd::Viewports vps{};
    cmd::Scissors scs{};
    float blend[4] = {1, 1, 1, 1};
    uint32_t stencil_ref = 0;
    cmd::Targets targets{};
    bool rendering = false;
    Attachments att;
    VkExtent2D extent{0, 0};
    VkPipeline bound = VK_NULL_HANDLE;
    bool set_bound[2] = {false, false};
    std::vector<Cb*>* out;  // earlier work of this submission (a memory wait sends it first)

    Builder(HostQueue& h, Cb* cc, std::vector<Cb*>* o = nullptr) : hq(h), c(cc), cb(cc->cb), out(o) {}

    // Work after a memory wait goes in a submission of its own, held at the queue's gate.
    void wait_value(const cmd::WaitValue& w) {
        if (!out || !hq.gate) {
            once("GPU waits on memory values are ignored here");
            return;
        }
        end_rendering();
        full_barrier(cb);
        Cb* next = get_cb(hq);
        if (!next) {
            once("a GPU wait on memory was ignored (no command buffer free)");
            return;
        }
        out->push_back(c);
        prepare_and_submit(hq, int(&hq - g_q), std::move(*out));
        out->clear();
        Gate gt;
        gt.w = w;
        push_gate(hq, gt);
        c = next, cb = next->cb;
        bound = VK_NULL_HANDLE;
        set_bound[0] = set_bound[1] = false;
    }

    void end_rendering() {
        if (rendering) vk::vkCmdEndRendering(cb), rendering = false;
    }

    bool begin_rendering() {
        VkRenderingAttachmentInfo color[8]{}, depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        att = {};
        uint32_t w = ~0u, h = ~0u;
        for (uint32_t i = 0; i < targets.count; ++i) {
            descriptor::Target t;
            color[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.color[i] = VK_FORMAT_UNDEFINED;
            if (!descriptor::target_info(targets.rtv[i], t)) continue;
            images::Image* im = images::of_resource(t.resource);
            if (!im) continue;
            auto f = images::facts(im);
            VkFormat vf;
            color[i].imageView = images::target_view(im, t.format, t.mip, t.layer, &vf);
            color[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            color[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
            color[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.color[i] = vf;
            w = std::min(w, std::max(1u, f.width >> t.mip));
            h = std::min(h, std::max(1u, f.height >> t.mip));
            images::gpu_wrote(im);
        }
        att.count = targets.count;
        // Trailing unbound slots need not be declared.
        while (att.count && att.color[att.count - 1] == VK_FORMAT_UNDEFINED) --att.count;
        descriptor::Target dt;
        bool has_depth = false;
        if (targets.dsv && descriptor::target_info(targets.dsv, dt)) {
            if (images::Image* im = images::of_resource(dt.resource)) {
                auto f = images::facts(im);
                if (f.depth) {
                    depth.imageView = images::target_view(im, 0, dt.mip, dt.layer);
                    depth.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                    depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
                    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                    att.depth = f.format;
                    att.stencil = f.stencil;
                    has_depth = true;
                    w = std::min(w, std::max(1u, f.width >> dt.mip));
                    h = std::min(h, std::max(1u, f.height >> dt.mip));
                }
            }
        }
        if (w == ~0u) {
            // Nothing to render into: the viewport's size (rasterisation still runs for UAV writes).
            w = vps.count ? uint32_t(std::max(1.0f, vps.v[0][2])) : 1;
            h = vps.count ? uint32_t(std::max(1.0f, vps.v[0][3])) : 1;
        }
        extent = {w, h};
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, extent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = att.count;
        ri.pColorAttachments = color;
        ri.pDepthAttachment = has_depth ? &depth : nullptr;
        ri.pStencilAttachment = has_depth && att.stencil ? &depth : nullptr;
        vk::vkCmdBeginRendering(cb, &ri);
        rendering = true;
        return true;
    }

    // The user SGPRs a program expects, from the bound root arguments.
    void sgprs(const shader::Program& p, const Bind& b, uint32_t* out) {
        for (auto& rp : p.layout) {
            if (rp.user_data >= 32) continue;
            const Arg& a = b.args[rp.slot & 63];
            uint32_t* o = out + rp.user_data;
            uint32_t n = std::min<uint32_t>(rp.sgprs ? rp.sgprs : 1, 32 - rp.user_data);
            switch (rp.root) {
                case shader::RootType::Constants:
                    for (uint32_t k = 0; k < n && k < 64; ++k) o[k] = a.c[k];
                    break;
                case shader::RootType::Table:
                    o[0] = uint32_t(a.value);
                    if (n > 1) o[1] = uint32_t(a.value >> 32);
                    break;
                case shader::RootType::Descriptor:
                    o[0] = uint32_t(a.value);
                    if (n > 1) o[1] = uint32_t(a.value >> 32);
                    if (a.value) memory::ensure(a.value);
                    break;
                case shader::RootType::ResourceHeap:
                    o[0] = uint32_t(view_heap);
                    if (n > 1) o[1] = uint32_t(view_heap >> 32);
                    break;
                case shader::RootType::SamplerHeap:
                    o[0] = uint32_t(sampler_heap);
                    if (n > 1) o[1] = uint32_t(sampler_heap >> 32);
                    break;
                case shader::RootType::ImmediateSampler: {
                    uint32_t w[4] = {};
                    if (root::static_sampler(b.root, rp.base_register, 0, w)) images::sampler(w);
                    for (uint32_t k = 0; k < n && k < 4; ++k) o[k] = w[k];
                    break;
                }
                default: break;
            }
        }
    }

    // The textures a program samples through tables: tell the image registry they are in use.
    void touch_images(const shader::Program& p, const Bind& b) {
        for (auto& u : p.used) {
            if (u.kind != shader::ResourceType::Srv || u.root != shader::RootType::Table) continue;
            uint64_t table = b.args[u.slot & 63].value;
            if (!table || (table >> 32)) continue;
            const uint32_t* w = reinterpret_cast<const uint32_t*>(table + uint64_t(u.table_offset_dwords) * 4);
            uint32_t type = w[3] >> 28;
            if (type < 8) continue;  // a buffer, not an image
            images::touch((uint64_t(w[0]) | uint64_t(w[1] & 0xff) << 32) << 8);
        }
    }

    void push(uint64_t rec) {
        uint64_t pc[4] = {rec, memory::page_table(), images::table(), 0};
        vk::vkCmdPushConstants(cb, g_layout, VK_SHADER_STAGE_ALL, 0, 32, pc);
    }

    // KEYLIME_TRACE_DRAWS=n: describes the first n draws (what they run, read and render into).
    static int trace_budget() {
        static int n = [] {
            char v[16]{};
            return GetEnvironmentVariableA("KEYLIME_TRACE_DRAWS", v, 16) ? atoi(v) : 0;
        }();
        return n;
    }
    void trace_draw(const char* why, pipeline::State* st, Compiled* k, const uint32_t* rec) {
        static std::atomic<int> done{0}, seen{0};
        static int skip = [] {
            char v[16]{};
            return GetEnvironmentVariableA("KEYLIME_TRACE_SKIP", v, 16) ? atoi(v) : 0;
        }();
        if (seen++ < skip) return;
        if (done.load() >= trace_budget()) return;
        ++done;
        uint64_t vh = 0, ph = 0;
        for (auto& s : st->shaders) {
            if (s.stage == 0) vh = s.hash;
            if (s.stage == 1) ph = s.hash;
            if (s.stage > 1) continue;
            char path[128];
            CreateDirectoryA("D:/keylime/logs/traced", nullptr);
            snprintf(path, sizeof path, "D:/keylime/logs/traced/%s-%016llx.bin", s.stage ? "ps" : "vs", (unsigned long long)s.hash);
            if (FILE* f = fopen(path, "wb")) fwrite(s.code.data(), 1, s.code.size(), f), fclose(f);
        }
        char sg[512] = {}, *o = sg;
        if (rec)
            for (int i = 0; i < 16; ++i) o += snprintf(o, 16, "%x ", rec[i]);
        char pg[512] = {}, *q = pg;
        if (rec)
            for (int i = 32; i < 48; ++i) q += snprintf(q, 16, "%x ", rec[i]);
        KL_INFO("draw", "%s: vs %016llx ps %016llx topo %u | %u targets (fmt %d, %ux%u) depth %d | vb0 %llx/%u ib %llx | vs sgprs %s| ps sgprs %s",
                why, (unsigned long long)vh, (unsigned long long)ph, topology, att.count, int(att.color[0]), extent.width, extent.height,
                int(att.depth), (unsigned long long)vb[0].address, vb[0].stride, (unsigned long long)ib.address, sg, pg);
        {
            const auto& bl = st->blend[0];
            KL_INFO("draw", "  blend0 enable %u src %u dst %u op %u | alpha src %u dst %u op %u | mask %x | rts %u fmt0 %u dsv %u | cull %u depth %u/%u/%u",
                    bl.enable, bl.src, bl.dst, bl.op, bl.src_alpha, bl.dst_alpha, bl.op_alpha, bl.write_mask, st->rt_count,
                    st->rt_formats[0], st->dsv_format, st->raster.cull, st->depth.depth_enable, st->depth.depth_write, st->depth.depth_func);
        }
        if (k && k->ps)
            for (auto& u : k->ps->used)
                if (u.root == shader::RootType::Table && u.kind == shader::ResourceType::Cbv) {
                    uint64_t table = bind[1].args[u.slot & 63].value;
                    if (!table) continue;
                    const uint32_t* w = reinterpret_cast<const uint32_t*>(table + uint64_t(u.table_offset_dwords) * 4);
                    uint64_t base = uint64_t(w[0]) | uint64_t(w[1] & 0xffff) << 32;
                    const float* c = reinterpret_cast<const float*>(base);
                    KL_INFO("draw", "  ps cb at %llx: %g %g %g %g | %g %g %g %g | %g %g %g %g", (unsigned long long)base, c[0], c[1], c[2], c[3],
                            c[4], c[5], c[6], c[7], c[8], c[9], c[10], c[11]);
                }
        for (auto& e : st->input)
            KL_INFO("draw", "  input %s%u format %u slot %u offset %u%s", e.semantic.c_str(), e.index, e.format, e.slot, e.offset,
                    e.per_instance ? " per instance" : "");
        if (k)
            for (auto& a : k->attrs) KL_INFO("draw", "  attribute location %u binding %u vk format %d offset %u", a.location, a.binding, int(a.format), a.offset);
        if (k && k->vs)
            for (auto& u : k->vs->used) {
                if (u.root != shader::RootType::Table) continue;
                uint64_t table = bind[1].args[u.slot & 63].value;
                if (!table) continue;
                const uint32_t* w = reinterpret_cast<const uint32_t*>(table + uint64_t(u.table_offset_dwords) * 4);
                uint64_t base = uint64_t(w[0]) | uint64_t(w[1] & 0xffff) << 32;
                KL_INFO("draw", "  vs reads kind %d slot %u +%u dwords %u: %08x %08x %08x %08x (base %llx stride %u records %u)", int(u.kind),
                        u.slot, u.table_offset_dwords, u.dwords, w[0], w[1], w[2], w[3], (unsigned long long)base, (w[1] >> 16) & 0x3fff, w[2]);
                if (base) {
                    const float* f = reinterpret_cast<const float*>(base);
                    KL_INFO("draw", "    data: %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g", f[0], f[1], f[2], f[3], f[4], f[5],
                            f[6], f[7], f[8], f[9], f[10], f[11], f[12], f[13], f[14], f[15]);
                }
            }
        if (k && k->vs)
            for (auto& in : k->vs->vs_inputs) KL_INFO("draw", "  vs input row %u mask %x -> v%u", in.row, in.mask, in.vgpr);
        if (vb[0].address) {
            const uint32_t* w = reinterpret_cast<const uint32_t*>(vb[0].address);
            char t[400], *o = t;
            for (int i = 0; i < 18; ++i) o += snprintf(o, 24, "%08x ", w[i]);
            KL_INFO("draw", "  vb0: %s", t);
        }
        if (k && k->vs)
            for (auto& rp : k->vs->layout)
                KL_INFO("draw", "  vs root: type %d slot %u sgpr %u n %u off %u", int(rp.root), rp.slot, rp.user_data, rp.sgprs, rp.table_offset_dwords);
        if (k && k->ps) {
            for (auto& rp : k->ps->layout)
                KL_INFO("draw", "  ps root: type %d slot %u sgpr %u n %u off %u", int(rp.root), rp.slot, rp.user_data, rp.sgprs, rp.table_offset_dwords);
            // The descriptors the PS reads from tables: what they name.
            for (auto& u : k->ps->used) {
                if (u.root != shader::RootType::Table) continue;
                uint64_t table = bind[1].args[u.slot & 63].value;
                if (!table) continue;
                const uint32_t* w = reinterpret_cast<const uint32_t*>(table + uint64_t(u.table_offset_dwords) * 4);
                uint64_t addr = (uint64_t(w[0]) | uint64_t(w[1] & 0xff) << 32) << 8;
                KL_INFO("draw", "  ps reads kind %d slot %u +%u: words %08x %08x %08x %08x -> %s", int(u.kind), u.slot, u.table_offset_dwords,
                        w[0], w[1], w[2], w[3], u.kind == shader::ResourceType::Srv ? images::describe(addr) : "");
            }
        }
    }

    bool draw_setup(bool indexed, bool mesh = false) {
        pipeline::State* st = bind[1].pso;
        if (!st || st->compute) return false;
        Compiled* k = obtain(st, true);
        if (!wait_budget([k] { return k->translated.load(std::memory_order_acquire); }, 0)) {
            dropped("draw");
            return false;
        }
        if (!k->ok) {
            trace_draw("skipped (pipeline not translated)", st, nullptr, nullptr);
            return false;
        }
        if (k->mesh != mesh) {
            once(mesh ? "a mesh dispatch with a vertex shader pipeline was skipped" : "a draw with a mesh shader pipeline was skipped");
            return false;
        }
        if (!mesh && (topology_class(topology) == ~0u || topology_class(topology) == 3)) {
            once("draws with topology %u are skipped (not supported yet)", topology);
            return false;
        }
        if (!rendering && !begin_rendering()) return false;
        VkPipeline p = variant(k, st, att, mesh ? 4 : topology);
        if (!p) return false;
        if (p != bound) vk::vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p), bound = p;
        if (!set_bound[1]) {
            VkDescriptorSet s = images::set();
            vk::vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, g_layout, 0, 1, &s, 0, nullptr);
            set_bound[1] = true;
        }
        uint64_t rec;
        uint32_t* r = record(c, RECORD_DWORDS, rec);
        if (!r) return false;
        memset(r, 0, RECORD_DWORDS * 4);
        sgprs(*k->vs, bind[1], r);
        if (k->ps) sgprs(*k->ps, bind[1], r + 32);
        if (k->ps) touch_images(*k->ps, bind[1]);
        touch_images(*k->vs, bind[1]);
        push(rec);
        trace_draw("draw", st, k, r);
        // Viewport (flipped: D3D's y points down in NDC terms), scissor.
        VkViewport v{0, float(extent.height), float(extent.width), -float(extent.height), 0, 1};
        if (vps.count) {
            const float* x = vps.v[0];
            v = {x[0], x[1] + x[3], x[2], -x[3], x[4], x[5]};
        }
        vk::vkCmdSetViewport(cb, 0, 1, &v);
        VkRect2D sc{{0, 0}, extent};
        if (scs.count) {
            const int32_t* s = scs.r[0];
            int32_t l = std::max(0, s[0]), t = std::max(0, s[1]);
            sc = {{l, t}, {uint32_t(std::max(0, s[2] - l)), uint32_t(std::max(0, s[3] - t))}};
        }
        vk::vkCmdSetScissor(cb, 0, 1, &sc);
        vk::vkCmdSetBlendConstants(cb, blend);
        vk::vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref);
        if (mesh) return true;
        vk::vkCmdSetPrimitiveTopology(cb, vk_topology(topology));
        bool strip = topology == 3 || topology == 5 || topology == 11 || topology == 13;
        vk::vkCmdSetPrimitiveRestartEnable(cb, indexed && strip && st->strip_cut ? VK_TRUE : VK_FALSE);
        // Vertex streams.
        for (auto& bd : k->binds) {
            const auto& view = vb[bd.binding];
            VkBuffer buf = g_zero.buffer;
            VkDeviceSize off = 0, stride = view.stride;
            uint64_t left;
            if (view.address && memory::ensure(view.address, 1)) memory::locate(view.address, buf, off, left);
            else stride = 0;
            vk::vkCmdBindVertexBuffers2(cb, bd.binding, 1, &buf, &off, nullptr, &stride);
        }
        if (indexed) {
            VkBuffer buf;
            VkDeviceSize off;
            uint64_t left;
            if (!ib.address || !memory::ensure(ib.address, 1) || !memory::locate(ib.address, buf, off, left)) {
                once("indexed draw without a usable index buffer (%llx) skipped", (unsigned long long)ib.address);
                return false;
            }
            vk::vkCmdBindIndexBuffer(cb, buf, off, ib.format == 42 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
        }
        return true;
    }

    Compiled* dispatch_setup() {
        pipeline::State* st = bind[0].pso;
        if (!st || !st->compute) return nullptr;
        bool pending;
        VkPipeline p = runnable(st, pending);
        if (!p) {
            if (pending) dropped("dispatch");
            else once("a dispatch whose pipeline could not be built was dropped");
            return nullptr;
        }
        Compiled* k = static_cast<Compiled*>(st->host);
        end_rendering();
        if (p != bound) vk::vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p), bound = p;
        if (!set_bound[0]) {
            VkDescriptorSet s = images::set();
            vk::vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_layout, 0, 1, &s, 0, nullptr);
            set_bound[0] = true;
        }
        uint64_t rec;
        uint32_t* r = record(c, RECORD_DWORDS, rec);
        if (!r) return nullptr;
        memset(r, 0, RECORD_DWORDS * 4);
        sgprs(*k->cs, bind[0], r);
        touch_images(*k->cs, bind[0]);
        push(rec);
        return k;
    }

    void mark(uint32_t kind, uint32_t x = 0, uint32_t y = 0, uint32_t z = 0) {
        const pipeline::State* st = bind[kind == 1 || kind == 7 ? 0 : 1].pso;
        checkpoint(cb, kind, st ? pipeline_key(st) : 0, x, y, z);
    }
    void dispatch(const cmd::Dispatch& d) {
        if (!dispatch_setup()) return;
        mark(1, d.x, d.y, d.z);
        struct After {  // a checkpoint behind the dispatch too: proves whether it finished
            Builder* b;
            ~After() { b->mark(7); }
        } after{this};
        if (uint64_t(d.x) * d.y * d.z > (1ull << 22)) {
            static std::atomic<int> n{0};
            if (n++ < 8) KL_WARN("gpu", "a dispatch of %u x %u x %u groups (pipeline %016llx)", d.x, d.y, d.z,
                                 (unsigned long long)pipeline_key(bind[0].pso));
        }
        vk::vkCmdDispatch(cb, d.x, d.y, d.z);
        stats::add(stats::DispatchesRun);
    }

    static uint64_t device_of(VkBuffer b, VkDeviceSize off) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = b;
        return vk::vkGetBufferDeviceAddress(dev(), &ai) + off;
    }

    void dispatch_mesh(const cmd::Dispatch& d) {
        uint32_t x = std::min(d.x, 65535u), y = std::min(d.y, 65535u), z = std::min(d.z, 65535u);
        if (!x || !y || !z) return;
        if (uint64_t(x) * y * z > (1u << 22)) {
            once("a mesh dispatch of %ux%ux%u groups was skipped (over the GPU's limit)", d.x, d.y, d.z);
            stats::add(stats::DrawsSkipped);
            return;
        }
        if (!draw_setup(false, true)) {
            stats::add(stats::DrawsSkipped);
            return;
        }
        mark(3);
        vk::vkCmdDrawMeshTasksEXT(cb, x, y, z);
        stats::add(stats::DrawsRun);
    }

    // ExecuteIndirect with a single Draw, DrawIndexed, Dispatch or DispatchMesh argument per command.
    void indirect(const cmd::Indirect& x) {
        static const uint32_t bytes[] = {16, 20, 12, 12};
        uint32_t stride = 0;
        std::vector<std::array<uint32_t, 4>> args;
        if (!misc::signature(x.signature, stride, args) || args.size() != 1) {
            once("ExecuteIndirect with %zu arguments per command is skipped (single-argument signatures run)", args.size());
            return;
        }
        uint32_t type = args[0][0], max = uint32_t(std::min<uint64_t>(x.max, 1u << 16));
        if (type == 105) type = 2;
        if (type == 10 && host::get().caps.mesh_shader) type = 3;
        if (type > 3) {
            once("ExecuteIndirect with argument type %u is skipped (not supported yet)", type);
            return;
        }
        if (!max) return;
        if (max > 1 && ((stride & 3) || stride < bytes[type])) {
            once("ExecuteIndirect with stride %u is skipped", stride);
            return;
        }
        VkBuffer ab, cbuf = VK_NULL_HANDLE;
        VkDeviceSize aoff, coff = 0;
        if ((x.args & 3) || !where(x.args, uint64_t(stride) * (max - 1) + bytes[type], ab, aoff)) {
            once("ExecuteIndirect with arguments the GPU cannot see (%llx) is skipped", (unsigned long long)x.args);
            return;
        }
        bool counted = x.count != 0;
        if (counted && ((x.count & 3) || !where(x.count, 4, cbuf, coff))) {
            once("ExecuteIndirect with a count the GPU cannot see (%llx) is skipped", (unsigned long long)x.count);
            return;
        }
        // Arguments go through a GPU pass first: past the count, or sizes no frame would use
        // (memory a dropped dispatch never wrote), a command becomes empty instead of a GPU hang.
        uint32_t words = bytes[type] / 4;
        uint64_t tight;
        uint32_t* r = g_clamp[type] ? record(c, max * words + 2, tight) : nullptr;
        if (!r) return;
        r[max * words] = max;
        r[max * words + 1] = 0;
        auto& blk = c->records[c->block];
        VkBuffer db = blk.buffer;
        VkDeviceSize doff = tight - blk.device;
        end_rendering();
        full_barrier(cb);
        vk::vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_clamp[type]);
        bound = g_clamp[type];
        struct {
            uint64_t src, dst, count;
            uint32_t stride, max;
        } pc{device_of(ab, aoff), tight, counted ? device_of(cbuf, coff) : tight + uint64_t(max) * words * 4, stride, max};
        vk::vkCmdPushConstants(cb, g_layout, VK_SHADER_STAGE_ALL, 0, 32, &pc);
        checkpoint(cb, 4, 0, (max + 63) / 64, 1, 1);  // keylime's indirect argument check
        vk::vkCmdDispatch(cb, (max + 63) / 64, 1, 1);
        full_barrier(cb);
        if (type == 2) {
            if (!dispatch_setup()) return;
            mark(1);
            for (uint32_t i = 0; i < max; ++i) vk::vkCmdDispatchIndirect(cb, db, doff + VkDeviceSize(i) * 12);
            stats::add(stats::DispatchesRun, max);
            return;
        }
        if (!draw_setup(type == 1, type == 3)) {
            stats::add(stats::DrawsSkipped);
            return;
        }
        mark(type == 3 ? 3 : 2);
        if (type == 3) vk::vkCmdDrawMeshTasksIndirectEXT(cb, db, doff, max, 12);
        else if (type == 0) vk::vkCmdDrawIndirect(cb, db, doff, max, 16);
        else vk::vkCmdDrawIndexedIndirect(cb, db, doff, max, 20);
        stats::add(stats::DrawsRun);
    }

    // A title buffer range as (buffer, offset); false if the GPU cannot see it.
    bool where(uint64_t a, uint64_t len, VkBuffer& b, VkDeviceSize& off) {
        uint64_t left;
        return a && memory::ensure(a, len) && memory::locate(a, b, off, left) && left >= len;
    }

    void write_value(const cmd::Write& w, bool is64) {
        if (!w.address) return;
        uint64_t v = w.value;
        if (is64 && (w.flags == 1 || w.flags == 2)) {
            LARGE_INTEGER q;
            QueryPerformanceCounter(&q);
            v = uint64_t(q.QuadPart);
        }
        VkBuffer b;
        VkDeviceSize off;
        if ((w.address & 3) == 0 && where(w.address, w.bytes, b, off)) {
            end_rendering();
            full_barrier(cb);
            vk::vkCmdUpdateBuffer(cb, b, off, w.bytes, &v);
            full_barrier(cb);
        } else if (w.bytes == 4) {
            *reinterpret_cast<volatile uint32_t*>(w.address) = uint32_t(v);
        } else {
            *reinterpret_cast<volatile uint64_t*>(w.address) = v;
        }
    }

    void copy_memory(uint64_t dst, uint64_t src, uint64_t len) {
        if (!dst || !src || !len) return;
        checkpoint(cb, 5, dst, uint32_t(len), 0, 0);
        VkBuffer db, sb;
        VkDeviceSize doff, soff;
        if (where(dst, len, db, doff) && where(src, len, sb, soff)) {
            end_rendering();
            full_barrier(cb);
            VkBufferCopy r{soff, doff, len};
            vk::vkCmdCopyBuffer(cb, sb, db, 1, &r);
            full_barrier(cb);
            images::written(dst, len);
        } else if (!safe_move(dst, src, len)) {
            once("a copy between unmapped title memory (%llx <- %llx) was dropped", (unsigned long long)dst, (unsigned long long)src);
        }
    }

    static void subresource(images::Image* im, uint32_t index, uint32_t& mip, uint32_t& layer) {
        auto f = images::facts(im);
        mip = index % f.mips, layer = index / f.mips;
    }

    void copy_texture(const cmd::CopyTexture& t) {
        end_rendering();
        checkpoint(cb, 6, 0);
        images::Image* di = t.dst.type == 0 ? images::of_resource(t.dst.resource) : nullptr;
        images::Image* si = t.src.type == 0 ? images::of_resource(t.src.resource) : nullptr;
        full_barrier(cb);
        if (di && si) {
            uint32_t dm, dl, sm, sl;
            subresource(di, uint32_t(t.dst.offset), dm, dl);
            subresource(si, uint32_t(t.src.offset), sm, sl);
            auto sf = images::facts(si), df = images::facts(di);
            VkImageCopy r{};
            r.srcSubresource = {VkImageAspectFlags(sf.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), sm, sl, 1};
            r.dstSubresource = {VkImageAspectFlags(df.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), dm, dl, 1};
            uint32_t sw = std::max(1u, sf.width >> sm), sh = std::max(1u, sf.height >> sm);
            if (t.has_box) r.srcOffset = {int32_t(t.box[0]), int32_t(t.box[1]), 0}, sw = t.box[3] - t.box[0], sh = t.box[4] - t.box[1];
            r.dstOffset = {int32_t(t.x), int32_t(t.y), 0};
            uint32_t dw = std::max(1u, df.width >> dm), dh = std::max(1u, df.height >> dm);
            r.extent = {std::min(sw, dw - std::min(dw, t.x)), std::min(sh, dh - std::min(dh, t.y)), 1};
            if (r.extent.width && r.extent.height)
                vk::vkCmdCopyImage(cb, sf.image, VK_IMAGE_LAYOUT_GENERAL, df.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            images::gpu_wrote(di);
        } else if (di && t.src.type == 1) {
            // Buffer footprint -> texture.
            resource::Info ri;
            if (!resource::info(t.src.resource, ri)) return;
            uint32_t bytes = 0, block = 1;
            texture::element(t.src.format, bytes, block);
            if (!bytes) return;
            uint32_t dm, dl;
            subresource(di, uint32_t(t.dst.offset), dm, dl);
            auto df = images::facts(di);
            uint64_t rows = (t.src.height + block - 1) / block;
            uint64_t len = uint64_t(t.src.row_pitch) * rows * std::max(1u, t.src.depth);
            VkBuffer b;
            VkDeviceSize off;
            if (!where(ri.address + t.src.offset, len, b, off)) return;
            VkBufferImageCopy r{};
            r.bufferOffset = off;
            r.bufferRowLength = t.src.row_pitch / bytes * block;
            r.bufferImageHeight = uint32_t(rows * block);
            r.imageSubresource = {VkImageAspectFlags(df.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), dm, dl, 1};
            r.imageOffset = {int32_t(t.x), int32_t(t.y), 0};
            uint32_t w = t.src.width, h = t.src.height;
            if (t.has_box) {
                r.bufferOffset += uint64_t(t.box[1] / block) * t.src.row_pitch + uint64_t(t.box[0] / block) * bytes;
                w = t.box[3] - t.box[0], h = t.box[4] - t.box[1];
            }
            uint32_t dw = std::max(1u, df.width >> dm), dh = std::max(1u, df.height >> dm);
            r.imageExtent = {std::min(w, dw - std::min(dw, t.x)), std::min(h, dh - std::min(dh, t.y)), 1};
            if (r.imageExtent.width && r.imageExtent.height)
                vk::vkCmdCopyBufferToImage(cb, b, df.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            images::gpu_wrote(di);
        } else if (si && t.dst.type == 1) {
            // Texture -> buffer footprint (readback).
            resource::Info ri;
            if (!resource::info(t.dst.resource, ri)) return;
            uint32_t bytes = 0, block = 1;
            texture::element(t.dst.format, bytes, block);
            if (!bytes) return;
            uint32_t sm, sl;
            subresource(si, uint32_t(t.src.offset), sm, sl);
            auto sf = images::facts(si);
            uint64_t rows = (t.dst.height + block - 1) / block;
            uint64_t len = uint64_t(t.dst.row_pitch) * rows;
            VkBuffer b;
            VkDeviceSize off;
            if (!where(ri.address + t.dst.offset, len, b, off)) return;
            VkBufferImageCopy r{};
            r.bufferOffset = off;
            r.bufferRowLength = t.dst.row_pitch / bytes * block;
            r.bufferImageHeight = uint32_t(rows * block);
            r.imageSubresource = {VkImageAspectFlags(sf.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), sm, sl, 1};
            uint32_t w = std::min(t.dst.width, std::max(1u, sf.width >> sm)), h = std::min(t.dst.height, std::max(1u, sf.height >> sm));
            r.imageExtent = {w, h, 1};
            vk::vkCmdCopyImageToBuffer(cb, sf.image, VK_IMAGE_LAYOUT_GENERAL, b, 1, &r);
        }
        full_barrier(cb);
    }

    void copy_resource(const cmd::CopyResource& r) {
        resource::Info d, s;
        if (!resource::info(r.dst, d) || !resource::info(r.src, s)) return;
        if (d.desc.dimension == resource::BUFFER && s.desc.dimension == resource::BUFFER) {
            copy_memory(d.address, s.address, std::min(d.desc.width, s.desc.width));
            return;
        }
        images::Image* di = images::of_resource(r.dst);
        images::Image* si = images::of_resource(r.src);
        if (!di || !si) return;
        end_rendering();
        full_barrier(cb);
        auto df = images::facts(di), sf = images::facts(si);
        std::vector<VkImageCopy> rs;
        for (uint32_t m = 0; m < std::min(df.mips, sf.mips); ++m) {
            VkImageCopy c{};
            VkImageAspectFlags a = df.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
            c.srcSubresource = {a, m, 0, std::min(df.depth_or_layers, sf.depth_or_layers)};
            c.dstSubresource = c.srcSubresource;
            c.extent = {std::max(1u, std::min(df.width, sf.width) >> m), std::max(1u, std::min(df.height, sf.height) >> m), 1};
            rs.push_back(c);
        }
        vk::vkCmdCopyImage(cb, sf.image, VK_IMAGE_LAYOUT_GENERAL, df.image, VK_IMAGE_LAYOUT_GENERAL, uint32_t(rs.size()), rs.data());
        images::gpu_wrote(di);
        full_barrier(cb);
    }

    void clear_rtv(const cmd::ClearRtv& c) {
        descriptor::Target t;
        if (!descriptor::target_info(c.handle, t)) return;
        images::Image* im = images::of_resource(t.resource);
        if (!im) return;
        end_rendering();
        auto f = images::facts(im);
        VkClearColorValue v;
        memcpy(v.float32, c.color, 16);
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, std::min(t.mip, f.mips - 1), 1, std::min(t.layer, f.depth_or_layers - 1), 1};
        full_barrier(cb);
        vk::vkCmdClearColorImage(cb, f.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        full_barrier(cb);
        images::gpu_wrote(im);
    }

    void clear_dsv(const cmd::ClearDsv& c) {
        descriptor::Target t;
        if (!descriptor::target_info(c.handle, t)) return;
        images::Image* im = images::of_resource(t.resource);
        if (!im) return;
        auto f = images::facts(im);
        if (!f.depth) return;
        end_rendering();
        VkImageAspectFlags a = 0;
        if (c.flags & 1) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if ((c.flags & 2) && f.stencil) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
        if (!a) return;
        VkClearDepthStencilValue v{c.depth, c.stencil};
        VkImageSubresourceRange r{a, std::min(t.mip, f.mips - 1), 1, std::min(t.layer, f.depth_or_layers - 1), 1};
        full_barrier(cb);
        vk::vkCmdClearDepthStencilImage(cb, f.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        full_barrier(cb);
    }

    void clear_uav(const cmd::ClearUav& c, bool is_float) {
        void* res = reinterpret_cast<void*>(c.resource);
        resource::Info ri;
        if (!resource::info(res, ri)) return;
        end_rendering();
        if (ri.desc.dimension == resource::BUFFER) {
            VkBuffer b;
            VkDeviceSize off;
            if (where(ri.address, ri.desc.width, b, off)) {
                full_barrier(cb);
                vk::vkCmdFillBuffer(cb, b, off, ri.desc.width & ~3ull, c.values[0]);
                full_barrier(cb);
            }
            return;
        }
        images::Image* im = images::of_resource(res);
        if (!im) return;
        auto f = images::facts(im);
        if (f.depth) return;
        VkClearColorValue v;
        memcpy(v.uint32, c.values, 16);
        (void)is_float;
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        full_barrier(cb);
        vk::vkCmdClearColorImage(cb, f.image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        full_barrier(cb);
        images::gpu_wrote(im);
    }

    void run(const Stream& s) {
        uint32_t index = 0;
        s.each([&](Op op, const uint8_t* p, uint32_t size) {
            auto get = [&](auto& v) { memcpy(&v, p, std::min<size_t>(sizeof v, size)); };
            static const bool every = [] { char v[8]{}; return GetEnvironmentVariableA("KEYLIME_CHECKPOINTS", v, 8) && !strcmp(v, "all"); }();
            if (every) checkpoint(cb, 8, uint64_t(op), index);  // KEYLIME_CHECKPOINTS=all: every command
            ++index;
            switch (op) {
                case Op::SetPipeline: {
                    uint64_t v;
                    get(v);
                    if (auto* st = pipeline::state(reinterpret_cast<void*>(v))) bind[st->compute ? 0 : 1].pso = st;
                    break;
                }
                case Op::RootSignature: {
                    cmd::Root r;
                    get(r);
                    bind[r.gfx & 1].root = reinterpret_cast<void*>(r.value);
                    break;
                }
                case Op::RootTable:
                case Op::RootView: {
                    cmd::Root r;
                    get(r);
                    bind[r.gfx & 1].args[r.slot & 63].value = r.value;
                    break;
                }
                case Op::RootConstants: {
                    cmd::RootConstants r{};
                    get(r);
                    Arg& a = bind[r.gfx & 1].args[r.slot & 63];
                    for (uint32_t k = 0; k < r.count && r.first + k < 64; ++k) a.c[r.first + k] = r.values[k];
                    break;
                }
                case Op::Heaps: {
                    cmd::Heaps h{};
                    get(h);
                    for (uint32_t i = 0; i < h.count; ++i) {
                        uint32_t kind;
                        uint64_t start;
                        if (!descriptor::heap_info(h.heaps[i], kind, start)) continue;
                        if (kind == 0) view_heap = start;
                        if (kind == 1) sampler_heap = start;
                    }
                    break;
                }
                case Op::IndexBuffer: get(ib); break;
                case Op::VertexBuffers: {
                    cmd::VertexBuffers v{};
                    get(v);
                    for (uint32_t i = 0; i < v.count && v.start + i < 32; ++i) vb[v.start + i] = v.views[i];
                    break;
                }
                case Op::Topology: {
                    uint64_t t;
                    get(t);
                    topology = uint32_t(t);
                    break;
                }
                case Op::Viewports: get(vps); break;
                case Op::Scissors: get(scs); break;
                case Op::BlendFactor: get(blend); break;
                case Op::StencilRef: {
                    uint64_t v;
                    get(v);
                    stencil_ref = uint32_t(v);
                    break;
                }
                case Op::Targets:
                    end_rendering();
                    targets = {};
                    get(targets);
                    break;
                case Op::ClearRtv: {
                    cmd::ClearRtv c;
                    get(c);
                    clear_rtv(c);
                    break;
                }
                case Op::ClearDsv: {
                    cmd::ClearDsv c;
                    get(c);
                    clear_dsv(c);
                    break;
                }
                case Op::ClearUavUint:
                case Op::ClearUavFloat: {
                    cmd::ClearUav c;
                    get(c);
                    clear_uav(c, op == Op::ClearUavFloat);
                    break;
                }
                case Op::Draw: {
                    cmd::Draw d;
                    get(d);
                    if (draw_setup(false)) {
                        mark(2);
                        vk::vkCmdDraw(cb, d.count, d.instances, d.first, d.first_instance);
                        stats::add(stats::DrawsRun);
                    } else {
                        stats::add(stats::DrawsSkipped);
                    }
                    break;
                }
                case Op::DrawIndexed: {
                    cmd::DrawIndexed d;
                    get(d);
                    if (draw_setup(true)) {
                        mark(2);
                        vk::vkCmdDrawIndexed(cb, d.count, d.instances, d.first, d.base, d.first_instance);
                        stats::add(stats::DrawsRun);
                    } else {
                        stats::add(stats::DrawsSkipped);
                    }
                    break;
                }
                case Op::Dispatch: {
                    cmd::Dispatch d;
                    get(d);
                    dispatch(d);
                    break;
                }
                case Op::Barrier:
                    end_rendering();
                    full_barrier(cb);
                    break;
                case Op::CopyBuffer:
                case Op::CopyMem: {
                    cmd::CopyBuffer c;
                    get(c);
                    copy_memory(c.dst, c.src, c.len);
                    break;
                }
                case Op::CopyTexture: {
                    cmd::CopyTexture t;
                    get(t);
                    copy_texture(t);
                    break;
                }
                case Op::CopyResource: {
                    cmd::CopyResource r;
                    get(r);
                    copy_resource(r);
                    break;
                }
                case Op::Write32:
                case Op::Write64: {
                    cmd::Write w;
                    get(w);
                    write_value(w, op == Op::Write64);
                    break;
                }
                case Op::ResolveQueries: {
                    cmd::ResolveQueries r;
                    get(r);
                    VkBuffer b;
                    VkDeviceSize off;
                    if (r.dst && (r.dst & 3) == 0 && where(r.dst, uint64_t(r.count) * 8, b, off)) {
                        end_rendering();
                        full_barrier(cb);
                        vk::vkCmdFillBuffer(cb, b, off, uint64_t(r.count) * 8, 0);
                        full_barrier(cb);
                    } else if (r.dst) {
                        memset(reinterpret_cast<void*>(r.dst), 0, size_t(r.count) * 8);
                    }
                    break;
                }
                case Op::ExecuteIndirect: {
                    cmd::Indirect x;
                    get(x);
                    indirect(x);
                    break;
                }
                case Op::DispatchMesh: {
                    cmd::Dispatch d;
                    get(d);
                    dispatch_mesh(d);
                    break;
                }
                case Op::WaitValue: {
                    cmd::WaitValue w;
                    get(w);
                    wait_value(w);
                    break;
                }
                default: break;
            }
        });
        end_rendering();
        full_barrier(cb);
    }
};

}  // namespace

void log_tdr() {
    auto read = [](const wchar_t* name, DWORD& v) {
        DWORD size = sizeof v;
        return RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers", name, RRF_RT_REG_DWORD,
                            nullptr, &v, &size) == ERROR_SUCCESS;
    };
    DWORD level = 3, delay = 2;
    bool has_level = read(L"TdrLevel", level), has_delay = read(L"TdrDelay", delay);
    KL_INFO("gpu", "Windows GPU timeout: level %lu%s, delay %lu s%s", (unsigned long)level, has_level ? "" : " (default)",
            (unsigned long)delay, has_delay ? "" : " (default)");
    if (level == 0) KL_WARN("gpu", "GPU timeout detection is off (TdrLevel 0): a GPU hang freezes the run instead of ending it");
}

bool start() {
    if (g_started) return true;
    auto& h = host::get();
    log_tdr();
    uint32_t fam[2] = {h.queues.graphics_family, h.queues.compute_family};
    VkQueue qs[2] = {h.queues.graphics, h.queues.compute};
    for (int i = 0; i < 2; ++i) {
        g_q[i].q = qs[i];
        g_q[i].family = fam[i];
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = fam[i];
        if (!KL_VK_CHECK(vk::vkCreateCommandPool(h.device, &pi, nullptr, &g_q[i].pool))) return false;
        VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        si.pNext = &ti;
        if (!KL_VK_CHECK(vk::vkCreateSemaphore(h.device, &si, nullptr, &g_q[i].timeline))) return false;
        if (KL_VK_CHECK(vk::vkCreateSemaphore(h.device, &si, nullptr, &g_q[i].gate)))
            std::thread([q = &g_q[i]] { gatekeeper(*q); }).detach();
        uint32_t nf = 0;
        vk::vkGetPhysicalDeviceQueueFamilyProperties(h.physical, &nf, nullptr);
        std::vector<VkQueueFamilyProperties> fp(nf);
        vk::vkGetPhysicalDeviceQueueFamilyProperties(h.physical, &nf, fp.data());
        if (fam[i] < nf && fp[fam[i]].timestampValidBits >= 36) {
            VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qi.queryCount = STAMPS;
            if (vk::vkCreateQueryPool(h.device, &qi, nullptr, &g_q[i].stamps) != VK_SUCCESS) g_q[i].stamps = VK_NULL_HANDLE;
        }
    }
    VkDescriptorSetLayout sl = images::set_layout();
    VkPushConstantRange pc{VK_SHADER_STAGE_ALL, 0, 32};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1;
    li.pSetLayouts = &sl;
    li.pushConstantRangeCount = 1;
    li.pPushConstantRanges = &pc;
    if (!KL_VK_CHECK(vk::vkCreatePipelineLayout(h.device, &li, nullptr, &g_layout))) return false;
    open_cache();
    open_used();
    for (uint32_t t = 0; t < 4; ++t) {
        auto code = shader::indirect_clamp(t);
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = code.size() * 4;
        mi.pCode = code.data();
        VkShaderModule m = VK_NULL_HANDLE;
        if (KL_VK_CHECK(vk::vkCreateShaderModule(h.device, &mi, nullptr, &m))) {
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
            ci.layout = g_layout;
            KL_VK_CHECK(vk::vkCreateComputePipelines(h.device, g_cache, 1, &ci, nullptr, &g_clamp[t]));
            vk::vkDestroyShaderModule(h.device, m, nullptr);
        }
    }
    if (!memory::make_block(1 << 16, false, g_zero)) return false;
    memset(g_zero.map, 0, size_t(g_zero.size));
    g_started = true;
    KL_INFO("gpu", "executor ready: title command lists run on %s (executor built %s %s; pipelines compile only on workers)",
            h.name.c_str(), __DATE__, __TIME__);
    return true;
}

void prepare(pipeline::State* st) {
    if (!g_started || !st || st->shaders.empty()) return;
    bool early;
    {
        std::lock_guard g(g_used_lock);
        early = g_used_before.count(pipeline_key(st)) != 0;
    }
    if (!early) return;
    ++g_started_early;
    obtain(st, false)->early = true;
}

void wanted(void* obj) {
    if (!g_started) return;
    pipeline::State* st = pipeline::state(obj);
    if (!st || !st->compute || st->shaders.empty()) return;
    Compiled* k = obtain(st, true);
    if (k->translated.load(std::memory_order_acquire) && k->ok && !k->compute.load() && !k->fast_claimed.load() &&
        !k->fast_queued.exchange(true))
        job([k] { fast_build(k); }, true);
}

// keylime precompile <dir>: builds every compute shader dumped there (KEYLIME_DUMP_SHADERS) into
// cache\spirv and the driver cache (cache\pipelines.bin) on all workers, as a run would, so the
// game finds them ready.
int precompile(const char* dir) {
    g_precompiling = true;
    std::vector<std::pair<std::string, uint64_t>> files;
    WIN32_FIND_DATAA fd;
    HANDLE f = FindFirstFileA((std::string(dir) + "/cs-*.bin").c_str(), &fd);
    if (f != INVALID_HANDLE_VALUE) {
        do {
            unsigned long long h = 0;
            if (sscanf(fd.cFileName, "cs-%llx.bin", &h) == 1) files.push_back({std::string(dir) + "\\" + fd.cFileName, uint64_t(h)});
        } while (FindNextFileA(f, &fd));
        FindClose(f);
    }
    printf("precompile: %zu compute shaders in %s\n", files.size(), dir);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Compiled*> all;
    for (auto& [path, h] : files) {
        FILE* in = fopen(path.c_str(), "rb");
        if (!in) continue;
        pipeline::Shader sh{5, {}, h};
        fseek(in, 0, SEEK_END);
        sh.code.resize(size_t(ftell(in)));
        fseek(in, 0, SEEK_SET);
        bool ok = fread(sh.code.data(), 1, sh.code.size(), in) == sh.code.size();
        fclose(in);
        if (!ok) continue;
        auto* st = new pipeline::State();
        st->compute = true;
        st->shaders.push_back(std::move(sh));
        all.push_back(obtain(st, true));
    }
    for (;;) {
        size_t done = 0;
        for (auto* k : all) done += k->ready.load();
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("precompile: %zu of %zu built, %.0f s\n", done, all.size(), sec);
        fflush(stdout);
        if (done == all.size()) break;
        Sleep(5000);
    }
    save_cache(true);
    save_used(true);
    printf("precompile: done in %.0f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    return 0;
}

// KEYLIME_ONE_QUEUE=1: compute lists run on the graphics queue too (in submission order).
int queue_for(uint32_t kind) {
    static const bool one = [] { char v[4]{}; return GetEnvironmentVariableA("KEYLIME_ONE_QUEUE", v, 4) && v[0] == '1'; }();
    return kind == 2 && !one ? COMPUTE : GRAPHICS;
}
VkQueue vk_queue(int q) { return g_q[q & 1].q; }

// Texture uploads and first-use layouts for what was just recorded go in a submission of their
// own ahead of it; work waits for the other queue's latest uploads too (images are shared).
namespace {
void read_stamps(HostQueue& h, int qi) {
    double ns = host::get().props.limits.timestampPeriod;
    while (!h.stamps_open.empty()) {
        uint64_t r[4];
        uint32_t s = h.stamps_open.front();
        if (vk::vkGetQueryPoolResults(dev(), h.stamps, s, 2, sizeof r, r, 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != VK_SUCCESS ||
            !r[1] || !r[3])
            break;
        h.stamps_open.pop_front();
        uint64_t a = std::max(r[0], h.stamp_last), b = r[2];
        if (b > a) {
            stats::add(qi ? stats::GpuUs1 : stats::GpuUs0, uint64_t(double(b - a) * ns / 1000.0));
            h.stamp_last = b;
        }
    }
}
void prepare_and_submit(HostQueue& h, int qi, std::vector<Cb*> cbs) {
    Cb* pre = get_cb(h);
    if (pre) checkpoint(pre->cb, 9, 0);
    if (pre && images::prepare(pre->cb, qi, h.submitted.load() + 1)) submit(h, {pre}, {}, {});
    else if (pre) cbs.insert(cbs.begin(), pre);  // empty: rides along
    std::vector<Semaphore> waits;
    int other = (qi & 1) ^ 1;
    uint64_t up = images::upload_value(other);
    if (up && g_q[other].timeline && up > completed(g_q[other])) waits.push_back({g_q[other].timeline, up});
    submit(h, cbs, waits, {});
}
}  // namespace

void execute(int qi, uint32_t count, void* const* lists) {
    HostQueue& h = g_q[qi & 1];
    {
        std::lock_guard g(h.lock);
        h.recording++;
        auto t0 = std::chrono::steady_clock::now();
        uint64_t waited0 = g_waited_total.load();
        std::vector<Cb*> cbs;
        int64_t stamp = -1;
        for (uint32_t i = 0; i < count; ++i) {
            Stream* s = list::stream_of(lists[i]);
            if (!s) continue;
            Cb* c = get_cb(h);
            if (!c) continue;
            if (stamp < 0 && h.stamps && h.stamps_open.size() < STAMPS / 2 - 1) {
                stamp = h.stamp_next;
                h.stamp_next = (h.stamp_next + 2) % STAMPS;
                vk::vkCmdResetQueryPool(c->cb, h.stamps, uint32_t(stamp), 2);
                vk::vkCmdWriteTimestamp2(c->cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, h.stamps, uint32_t(stamp));
            }
            Builder b(h, c, &cbs);
            b.run(*s);
            cbs.push_back(b.c);
        }
        if (stamp >= 0 && !cbs.empty()) {
            vk::vkCmdWriteTimestamp2(cbs.back()->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, h.stamps, uint32_t(stamp) + 1);
            h.stamps_open.push_back(uint32_t(stamp));
        }
        uint64_t spent = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
        uint64_t waited = g_waited_total.load() - waited0;
        stats::add(stats::RecordUs, spent > waited ? spent - waited : 0);
        if (!cbs.empty()) prepare_and_submit(h, qi, std::move(cbs));
        read_stamps(h, qi);
        h.recording--;
    }
    collect();
}

void kick() { kick_held(); }


void save_lists(bool with_cache) {
    save_used(true);
    if (with_cache) save_cache(true);
}

void signal(int qi, VkSemaphore s, uint64_t v) {
    HostQueue& h = g_q[qi & 1];
    {
        std::lock_guard g(h.lock);
        submit(h, {}, {}, {{s, v}});
    }
    collect();
}

void wait(int qi, VkSemaphore s, uint64_t v, bool signal_submitted) {
    HostQueue& h = g_q[qi & 1];
    std::lock_guard g(h.lock);
    if (signal_submitted || !h.gate) {
        submit(h, {}, {{s, v}}, {});
        return;
    }
    Gate gt;
    gt.sem = s, gt.sem_value = v;
    push_gate(h, gt);
}

void dump_state() {
    uint64_t now = GetTickCount64();
    for (int i = 0; i < 2; ++i) {
        HostQueue& h = g_q[i];
        if (!h.q) continue;
        uint64_t gate_at = 0;
        if (h.gate) vk::vkGetSemaphoreCounterValue(dev(), h.gate, &gate_at);
        std::vector<Gate> gs;
        {
            std::lock_guard l(h.gates_lock);
            gs.assign(h.gates.begin(), h.gates.end());
        }
        KL_INFO("hang", "queue %d: submitted %llu, GPU done %llu, gate %llu of %llu, %zu gates waiting, %llu opened late",
                i, (unsigned long long)h.submitted.load(), (unsigned long long)completed(h), (unsigned long long)gate_at,
                (unsigned long long)h.gate_value, gs.size(), (unsigned long long)h.opened_late.load());
        for (size_t k = 0; k < gs.size() && k < 4; ++k) {
            auto& g = gs[k];
            if (g.sem) {
                uint64_t v = 0;
                vk::vkGetSemaphoreCounterValue(dev(), g.sem, &v);
                KL_INFO("hang", "  gate %llu (%llu ms): fence wants %llu, at %llu", (unsigned long long)g.opens,
                        (unsigned long long)(now - g.since), (unsigned long long)g.sem_value, (unsigned long long)v);
            } else {
                bool ok;
                uint64_t v = read_value(g.w, ok);
                KL_INFO("hang", "  gate %llu (%llu ms): memory %llx cmp %llu wants %llx, holds %llx%s", (unsigned long long)g.opens,
                        (unsigned long long)(now - g.since), (unsigned long long)g.w.address, (unsigned long long)g.w.compare,
                        (unsigned long long)g.w.reference, (unsigned long long)v, ok ? "" : " (unreadable)");
            }
        }
    }
    size_t jobs;
    {
        std::lock_guard g(g_jobs_lock);
        jobs = g_jobs.size() + g_urgent.size() + g_warm.size();
        if (!g_urgent.empty()) KL_INFO("hang", "pipelines: %zu urgent compile jobs queued", g_urgent.size());
    }
    KL_INFO("hang", "pipelines: %zu compile jobs queued, %llu compiling, %llu built, %llu draws or dispatches dropped waiting",
            jobs, (unsigned long long)stats::g_counters[stats::Compiling].load(),
            (unsigned long long)stats::g_counters[stats::PipelinesBuilt].load(),
            (unsigned long long)stats::g_counters[stats::WaitingOnCompile].load());
    KL_INFO("hang", "submissions: %zu held until their waits are signalled (%llu held this run)", g_pending_n.load(),
            (unsigned long long)g_held.load());
    std::vector<Event> ev;
    {
        std::lock_guard g(g_ev_lock);
        uint64_t n = g_ev_n < 64 ? g_ev_n : 64;
        for (uint64_t k = g_ev_n - n; k < g_ev_n; ++k) ev.push_back(g_ev[k % 64]);
    }
    KL_INFO("hang", "last %zu queue events:", ev.size());
    for (auto& e : ev) KL_INFO("hang", "  -%llu ms %s", (unsigned long long)(now - e.ms), e.text);
}

void device_lost(const char* where) {
    static std::atomic<bool> once{false};
    if (!once.exchange(true)) {
        KL_ERROR("gpu", "the GPU stopped (Vulkan device lost while %s): a shader or command hung the GPU, so this run ends here", where);
        if (host::get().caps.device_fault && vk::vkGetDeviceFaultInfoEXT) {
            VkDeviceFaultCountsEXT c{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
            if (vk::vkGetDeviceFaultInfoEXT(dev(), &c, nullptr) == VK_SUCCESS || true) {
                std::vector<VkDeviceFaultAddressInfoEXT> addr(c.addressInfoCount);
                std::vector<VkDeviceFaultVendorInfoEXT> vend(c.vendorInfoCount);
                c.vendorBinarySize = 0;
                VkDeviceFaultInfoEXT fi{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
                fi.pAddressInfos = addr.data(), fi.pVendorInfos = vend.data();
                vk::vkGetDeviceFaultInfoEXT(dev(), &c, &fi);
                KL_ERROR("gpu", "  device fault: %s (%u addresses, %u vendor records)", fi.description, c.addressInfoCount, c.vendorInfoCount);
                static const char* types[] = {"none", "read", "write", "execute", "instruction pointer unknown", "instruction pointer invalid", "instruction pointer fault"};
                for (auto& a : addr)
                    KL_ERROR("gpu", "    %s at %016llx (+-%llx)", types[a.addressType < 7 ? a.addressType : 0], (unsigned long long)a.reportedAddress,
                             (unsigned long long)a.addressPrecision);
                for (auto& v : vend) KL_ERROR("gpu", "    vendor: %s (code %llx, data %llx)", v.description, (unsigned long long)v.vendorFaultCode, (unsigned long long)v.vendorFaultData);
            }
        }
        if (host::get().caps.checkpoints && vk::vkGetQueueCheckpointDataNV)
            for (int qi = 0; qi < 2; ++qi) {
                if (!g_q[qi].q) continue;
                uint32_t n = 0;
                vk::vkGetQueueCheckpointDataNV(g_q[qi].q, &n, nullptr);
                std::vector<VkCheckpointDataNV> cp(n, {VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV});
                vk::vkGetQueueCheckpointDataNV(g_q[qi].q, &n, cp.data());
                static const char* kinds[] = {"?", "dispatch", "draw", "mesh draw", "indirect check", "memory copy", "texture copy", "after dispatch", "list command", "texture preparation"};
                for (uint32_t k = 0; k < n; ++k) {
                    uint64_t m = reinterpret_cast<uint64_t>(cp[k].pCheckpointMarker);
                    if (!m || m > MARKS) continue;
                    const Mark& mk = g_marks[m - 1];
                    KL_ERROR("gpu", "  queue %d last checkpoint at stage %x: %s of pipeline %016llx, %u x %u x %u groups%s", qi, cp[k].stage,
                             kinds[mk.kind < 10 ? mk.kind : 0], (unsigned long long)mk.key, mk.x, mk.y, mk.z, mk.x ? "" : " (indirect or draw)");
                }
            }
        dump_state();
        TerminateProcess(GetCurrentProcess(), 0xDEAD0005);
    }
    for (;;) Sleep(1000);
}

bool run(int qi, const std::function<void(VkCommandBuffer)>& f, VkSemaphore wait_bin, VkSemaphore signal_bin) {
    HostQueue& h = g_q[qi & 1];
    std::lock_guard g(h.lock);
    struct Recording {
        HostQueue& h;
        Recording(HostQueue& q) : h(q) { h.recording++; }
        ~Recording() { h.recording--; }
    } rec(h);
    std::vector<Cb*> cbs;
    Cb* pre = get_cb(h);
    if (pre && images::prepare(pre->cb, qi, h.submitted.load() + 1)) submit(h, {pre}, {}, {});
    else if (pre) cbs.push_back(pre);  // empty: rides along
    Cb* c = get_cb(h);
    if (!c) return false;
    cbs.push_back(c);
    f(c->cb);
    std::vector<Semaphore> w, s;
    if (wait_bin) w.push_back({wait_bin, 0});
    if (signal_bin) s.push_back({signal_bin, 0});
    int other = (qi & 1) ^ 1;
    uint64_t up = images::upload_value(other);
    if (up && g_q[other].timeline && up > completed(g_q[other])) w.push_back({g_q[other].timeline, up});
    auto p = submit(h, cbs, w, s);
    std::unique_lock l(g_sub_lock);
    while (!g_sub_cv.wait_for(l, std::chrono::seconds(5), [&] { return p->sent; }))
        KL_WARN("gpu", "presenting waits for queue %d's held work (%zu submissions held)", qi & 1, g_pending.size());
    return true;
}

void with_queue(int qi, const std::function<void(VkQueue)>& f) {
    HostQueue& h = g_q[qi & 1];
    std::lock_guard g(h.lock);
    std::lock_guard q(h.qlock);
    f(h.q);
}

}  // namespace exec

void retire(std::function<void()> f) {
    std::lock_guard g(exec::g_retire_lock);
    // +2: what is being recorded now may go as two submissions (its uploads, then its work).
    exec::g_retirees.push_back({{exec::g_q[0].submitted.load() + 2, exec::g_q[1].submitted.load() + 2}, std::move(f)});
}

}  // namespace kl::gpu
