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
#include <utility>
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
#include "gpu/exec/queues.inc"
#include "gpu/exec/pipelines.inc"
#include "gpu/exec/builder.inc"
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
            qi.queryCount = PROF;
            if (profiling() && vk::vkCreateQueryPool(h.device, &qi, nullptr, &g_q[i].prof) == VK_SUCCESS) {
                vk::vkResetQueryPool(h.device, g_q[i].prof, 0, PROF);
                KL_INFO("gpu", "profiling: GPU time per pipeline is logged every 10 s (KEYLIME_PROFILE)");
            } else {
                g_q[i].prof = VK_NULL_HANDLE;
            }
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

// keylime precompile <dir>: builds every compute shader in <dir> (any *.bin: the game's whole
// shader pack from tools/extract_shaders.py, or KEYLIME_DUMP_SHADERS dumps) into cache\spirv and
// the driver cache (cache\pipelines.bin), keyed as a run keys them, so the game finds them ready.
// Shaders already built are skipped, so it can be stopped and started again.
int precompile(const char* dir) {
    g_precompiling = true;
    std::vector<std::string> files;
    WIN32_FIND_DATAA fd;
    HANDLE f = FindFirstFileA((std::string(dir) + "/*.bin").c_str(), &fd);
    if (f != INVALID_HANDLE_VALUE) {
        do files.push_back(std::string(dir) + "\\" + fd.cFileName);
        while (FindNextFileA(f, &fd));
        FindClose(f);
    }
    printf("precompile: %zu shader files in %s\n", files.size(), dir);
    auto t0 = std::chrono::steady_clock::now();
    std::vector<Compiled*> all;
    size_t other = 0;
    for (auto& path : files) {
        FILE* in = fopen(path.c_str(), "rb");
        if (!in) continue;
        pipeline::Shader sh{5, {}, 0};
        _fseeki64(in, 0, SEEK_END);
        sh.code.resize(size_t(_ftelli64(in)));
        _fseeki64(in, 0, SEEK_SET);
        bool ok = !sh.code.empty() && fread(sh.code.data(), 1, sh.code.size(), in) == sh.code.size();
        fclose(in);
        if (!ok) continue;
        // Compute programs only (draw pipelines need the title's state, and build in milliseconds).
        shader::Container c;
        std::string err;
        if (!shader::parse(sh.code.data(), sh.code.size(), c, err) || !find(c, shader::HwStage::Cs)) {
            ++other;
            continue;
        }
        uint64_t h = 0xcbf29ce484222325ull;  // the runtime's key for these bytes (pipeline.cpp)
        for (uint8_t b : sh.code) h = (h ^ b) * 0x100000001b3ull;
        sh.hash = h;
        auto* st = new pipeline::State();
        st->compute = true;
        st->shaders.push_back(std::move(sh));
        all.push_back(obtain(st, true));
    }
    printf("precompile: %zu compute shaders queued (%zu files are other stages)\n", all.size(), other);
    for (int tick = 1;; ++tick) {
        size_t done = 0;
        for (auto* k : all) done += k->ready.load();
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("precompile: %zu of %zu built, %.0f s\n", done, all.size(), sec);
        fflush(stdout);
        if (done == all.size()) break;
        if (tick % 60 == 0) save_cache(true);  // every 5 minutes: a stopped run keeps what it built
        Sleep(5000);
    }
    save_cache(true);
    // KEYLIME_PRECOMPILE_OPTIMIZED=1: the optimized builds too (runs use them from the start
    // instead of the quick ones; hours for the whole game, so an overnight job). Resumable.
    char opt[4]{};
    if (GetEnvironmentVariableA("KEYLIME_PRECOMPILE_OPTIMIZED", opt, 4) && opt[0] == '1') {
        std::atomic<size_t> left{0};
        for (auto* k : all) {
            if (!k->ok || !k->csm) continue;
            ++left;
            job([k, &left] {
                if (!cached_optimized(k)) optimize(k);
                --left;
            }, true);
        }
        size_t total = left.load();
        printf("precompile: %zu optimized builds queued\n", total);
        for (int tick = 1; left.load(); ++tick) {
            Sleep(5000);
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("precompile: %zu of %zu optimized, %.0f s\n", total - left.load(), total, sec);
            fflush(stdout);
            if (tick % 60 == 0) save_cache(true);
        }
        save_cache(true);
    }
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
// Charges each draw's or dispatch's GPU time to its pipeline; every 10 s logs where the time went.
struct ProfEntry {
    double us = 0;
    uint64_t n = 0;
};
std::mutex g_prof_lock;
std::unordered_map<uint64_t, ProfEntry> g_prof;  // pipeline key ^ kind << 60
void read_profile(HostQueue& h) {
    if (!h.prof) return;
    double ns = host::get().props.limits.timestampPeriod;
    auto result = [&](uint32_t slot, uint64_t& t) {
        uint64_t r[2];
        if (vk::vkGetQueryPoolResults(dev(), h.prof, slot, 1, sizeof r, r, 16, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != VK_SUCCESS || !r[1])
            return false;
        t = r[0];
        return true;
    };
    while (h.prof_open.size() >= 2) {
        auto a = h.prof_open[0], z = h.prof_open[1];
        uint64_t ta, tz;
        if (!result(a.slot, ta) || !result(z.slot, tz)) break;
        h.prof_open.pop_front();
        vk::vkResetQueryPool(dev(), h.prof, a.slot, 1);
        if (a.kind && !z.kind && a.list == z.list && tz > ta) {
            std::lock_guard g(g_prof_lock);
            auto& e = g_prof[a.key ^ (uint64_t(a.kind) << 60)];
            e.us += double(tz - ta) * ns / 1000.0;
            e.n++;
        }
    }
    static std::atomic<int64_t> last{0};
    int64_t now = int64_t(GetTickCount64()), was = last.load();
    if (now - was < 10000 || !last.compare_exchange_strong(was, now)) return;
    std::vector<std::pair<uint64_t, ProfEntry>> all;
    {
        std::lock_guard g(g_prof_lock);
        all.assign(g_prof.begin(), g_prof.end());
        g_prof.clear();
    }
    double total = 0;
    for (auto& e : all) total += e.second.us;
    std::sort(all.begin(), all.end(), [](auto& x, auto& y) { return x.second.us > y.second.us; });
    static const char* kinds[] = {"", "dispatch", "draw", "mesh draw"};
    KL_INFO("gpu", "profile: %.0f ms of GPU time in 10 s over %zu pipelines; the most:", total / 1000, all.size());
    for (size_t i = 0; i < all.size() && i < 15; ++i) {
        uint64_t kind = all[i].first >> 60, key = all[i].first & ((1ull << 60) - 1);
        KL_INFO("gpu", "profile:   %6.0f ms (%4.1f%%) %s %015llx, %llu times, %.0f us each", all[i].second.us / 1000,
                100 * all[i].second.us / std::max(total, 1.0), kind < 4 ? kinds[kind] : "?", (unsigned long long)key,
                (unsigned long long)all[i].second.n, all[i].second.us / double(all[i].second.n));
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
            h.prof_list++;
            b.run(*s);
            b.stamp(0, 0);
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
        read_profile(h);
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
