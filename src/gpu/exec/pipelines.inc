// keylime: pipelines: translated shader modules, the shader and pipeline caches, compile workers, frame wait budgets, draw pipeline variants (part of the command executor; included by exec.cpp).
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
// waits for it within a budget per frame (BUDGET_US); past it, the draw or dispatch is skipped. A compute
// pipeline gets a quick unoptimized build first (well under a second), then the optimized one. Pipelines an earlier run used (listed in
// cache\used-pipelines.txt) start building as soon as the title creates them.
std::mutex g_jobs_lock, g_done_lock;
std::condition_variable g_jobs_cv, g_done_cv;
// g_urgent: what a draw or dispatch waits for now (first-use translations, quick compute
// builds). Some workers take only those, so a backlog of optimized builds (30-90 s each for
// big translated compute shaders) can never hold up a dispatch the title is waiting on.
// g_warm: early starts (translation and quick build of pipelines earlier runs used), taken by
// every worker whenever nothing urgent waits, so they are ready before the scene needs them;
// g_jobs: optimized builds, on two background workers, and on any idle non-strict worker at lowest priority.
std::deque<std::function<void()>> g_jobs, g_urgent, g_warm;
void queue_gauges() {
    stats::set(stats::CompileQueue, g_jobs.size() + g_urgent.size() + g_warm.size());
    stats::set(stats::CompileUrgent, g_urgent.size());
    stats::set(stats::CompileOptimized, g_jobs.size());
    stats::set(stats::CompileWarm, g_warm.size());
}
void notify() {
    { std::lock_guard g(g_done_lock); }
    g_done_cv.notify_all();
}

// Driver pipeline cache, kept in cache\pipelines.bin across runs: written at most every 2 minutes,
// when nothing urgent or early is queued, and on a hang (the copy of a 3 GB cache stalls the workers).
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
    if (built == saved_at || (!force && now - last < std::chrono::minutes(2))) return;
    if (!force) {
        std::lock_guard j(g_jobs_lock);
        if (!g_urgent.empty() || !g_warm.empty()) return;
    }
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
        // Background (optimized) builds: 2 workers, plus up to a quarter of the cores while idle, at lowest priority.
        unsigned cores = std::max(4u, std::thread::hardware_concurrency());
        // `keylime precompile`: KEYLIME_PRECOMPILE_WORKERS driver compiles at once (default 8; each
        // big shader takes gigabytes of driver memory, and more ran a 32 GB machine out of it).
        unsigned pre = [] { char v[8]{}; return GetEnvironmentVariableA("KEYLIME_PRECOMPILE_WORKERS", v, 8) ? unsigned(std::max(2, atoi(v))) : 8u; }();
        unsigned n = g_precompiling ? pre / 2 : 2;
        const unsigned URGENT_ONLY = g_precompiling ? pre - pre / 2 : std::max(4u, std::min(16u, cores / 2));
        // Kinds of worker: 4 strictly urgent (never busy with a long build when a dispatch needs
        // one), the rest of the urgent pool urgent then warm, and n background workers urgent,
        // warm, then optimized builds.
        constexpr unsigned STRICT_URGENT = 4;
        static unsigned borrowed = 0;
        const unsigned BORROW = std::max(2u, cores / 4);
        g_backlog_at = 2 * URGENT_ONLY;
        // During a backlog a few more workers take urgent builds at lowest priority (the title mostly waits then).
        for (unsigned i = 0; i < (g_precompiling ? 0 : cores / 6); ++i)
            std::thread([] {
                SetThreadDescription(GetCurrentThread(), L"keylime-compile-burst");
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                for (;;) {
                    std::function<void()> f;
                    {
                        std::unique_lock g(g_jobs_lock);
                        g_jobs_cv.wait(g, [] { return g_urgent.size() > g_backlog_at; });
                        f = std::move(g_urgent.front());
                        g_urgent.pop_front();
                        queue_gauges();
                    }
                    f();
                    stats::g_counters[stats::Compiling].fetch_sub(1);
                    notify();
                    save_cache();
                    save_used();
                }
            }).detach();
        for (unsigned i = 0; i < n + URGENT_ONLY; ++i)
            std::thread([urgent_only = i >= n, strict = i >= n + URGENT_ONLY - std::min(STRICT_URGENT, URGENT_ONLY), BORROW] {
                SetThreadDescription(GetCurrentThread(), strict ? L"keylime-compile-strict" : urgent_only ? L"keylime-compile-urgent" : L"keylime-compile");
                SetThreadPriority(GetCurrentThread(), strict ? THREAD_PRIORITY_ABOVE_NORMAL : urgent_only ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_LOWEST);
                for (;;) {
                    std::function<void()> f;
                    bool background = false;
                    {
                        std::unique_lock g(g_jobs_lock);
                        auto jobs_ok = [&] { return !g_jobs.empty() && (!urgent_only || borrowed < BORROW); };
                        g_jobs_cv.wait(g, [&] { return !g_urgent.empty() || (!strict && (!g_warm.empty() || jobs_ok())); });
                        auto& from = !g_urgent.empty() ? g_urgent : !g_warm.empty() ? g_warm : g_jobs;
                        background = &from == &g_jobs && urgent_only;
                        if (background) ++borrowed;
                        f = std::move(from.front());
                        from.pop_front();
                        queue_gauges();
                    }
                    if (background) SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
                    f();
                    if (background) {
                        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
                        {
                            std::lock_guard g(g_jobs_lock);
                            --borrowed;
                        }
                        g_jobs_cv.notify_all();
                    }
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
        queue_gauges();
    }
    // notify_one could wake an urgent-only worker for normal work and lose the wakeup.
    g_jobs_cv.notify_all();
}
// Per shown frame, one window shared by all waiting threads: KEYLIME_DRAW_WAIT_MS (4000), KEYLIME_DISPATCH_WAIT_MS (12000).
// With more urgent builds queued than the workers clear in a few seconds, a wait rarely ends in
// time, so frames wait only KEYLIME_BACKLOG_DRAW_WAIT_MS (10) and KEYLIME_BACKLOG_DISPATCH_WAIT_MS (30).
uint64_t budget_ms(const char* name, uint64_t fallback) {
    char v[16]{};
    return GetEnvironmentVariableA(name, v, 16) ? uint64_t(atoi(v)) : fallback;
}
const uint64_t BUDGET_US[2] = {budget_ms("KEYLIME_DRAW_WAIT_MS", 4000) * 1000, budget_ms("KEYLIME_DISPATCH_WAIT_MS", 12000) * 1000};
const uint64_t BACKLOG_US[2] = {budget_ms("KEYLIME_BACKLOG_DRAW_WAIT_MS", 10) * 1000, budget_ms("KEYLIME_BACKLOG_DISPATCH_WAIT_MS", 30) * 1000};
struct FrameWait {
    std::mutex m;
    uint64_t frame = ~0ull, spent_us = 0;
    unsigned waiting = 0;
    std::chrono::steady_clock::time_point since;
};
FrameWait g_frame_wait[2];
std::atomic<uint64_t> g_waited_total{0};
template <class F>
bool wait_budget(F done, int kind) {
    if (done()) return true;
    using clock = std::chrono::steady_clock;
    auto us_since = [](clock::time_point t) { return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - t).count()); };
    uint64_t frame = stats::g_counters[stats::FramesShown].load(std::memory_order_relaxed);
    auto& w = g_frame_wait[kind];
    uint64_t left;
    {
        std::lock_guard g(w.m);
        if (w.frame != frame) w.frame = frame, w.spent_us = 0, w.waiting = 0;
        uint64_t used = w.spent_us + (w.waiting ? us_since(w.since) : 0), budget = backlogged() ? BACKLOG_US[kind] : BUDGET_US[kind];
        if (used >= budget) return false;
        left = budget - used;
        if (!w.waiting++) w.since = clock::now();
    }
    auto t0 = clock::now();
    bool ok;
    {
        std::unique_lock g(g_done_lock);
        ok = g_done_cv.wait_until(g, t0 + std::chrono::microseconds(left), done);
    }
    uint64_t us = us_since(t0);
    {
        std::lock_guard g(w.m);
        if (w.frame == frame && w.waiting && !--w.waiting) w.spent_us += us_since(w.since);
    }
    g_waited_total += us;
    stats::add(stats::CompileWaitUs, us);
    return ok;
}
void dropped(const char* what) {
    static std::atomic<bool> told{false};
    if (!told.exchange(true))
        KL_INFO("gpu", "draws whose pipeline is not ready within %llu ms a frame, and dispatches past %llu ms a frame, are skipped meanwhile "
                       "(%llu and %llu ms while more than %llu urgent builds are queued)",
                (unsigned long long)(BUDGET_US[0] / 1000), (unsigned long long)(BUDGET_US[1] / 1000), (unsigned long long)(BACKLOG_US[0] / 1000),
                (unsigned long long)(BACKLOG_US[1] / 1000), (unsigned long long)g_backlog_at.load());
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

enum class Build { Optimized, Urgent, Rebuild };
void build_variant(Compiled* k, pipeline::State* st, const Attachments& a, uint32_t topo, uint64_t key, Build how = Build::Optimized);
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
            job([k, st, a, topo, key, urgent] { build_variant(k, st, a, topo, key, urgent ? Build::Urgent : Build::Optimized); }, urgent, urgent, !urgent);  // early ones: warm tier
        } else if (urgent && k->pending.count(key) && k->urgent_variants.insert(key).second) {
            // An early start asked for it at low priority (behind thousands of background builds): a
            // draw waits now, so it gets an urgent build of its own; the first one done is kept.
            job([k, st, a, topo, key] { build_variant(k, st, a, topo, key, Build::Urgent); }, true, true);
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

// During a build backlog a draw's variant is built unoptimized (several times faster in the
// driver), and the optimized one replaces it later from the background queue.
void build_variant(Compiled* k, pipeline::State* st, const Attachments& a, uint32_t topo, uint64_t key, Build how) {
    if (how != Build::Rebuild) {
        std::lock_guard g(st->lock);
        if (!k->pending.count(key)) return;  // the other (urgent or early) build got there first
    }
    bool quick = how == Build::Urgent && backlogged() && !g_precompiling;
    int expand = topo == RECTLIST ? 0 : topo == QUADLIST ? 1 : -1;
    auto finish = [&](VkPipeline p) {
        if (how == Build::Rebuild) {
            if (!p) return;
            VkPipeline old;
            {
                std::lock_guard g(st->lock);
                old = std::exchange(k->variants[key], p);
            }
            if (old) retire([old] { vk::vkDestroyPipeline(dev(), old, nullptr); });
            return;
        }
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
        if (quick && p) job([k, st, a, topo, key] { build_variant(k, st, a, topo, key, Build::Rebuild); });
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
    if (quick) ci.flags |= VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION_BIT;
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

