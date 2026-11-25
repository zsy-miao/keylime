#include "stats.h"

#include <windows.h>

#include <psapi.h>

#include <string>
#include <thread>

#include "log.h"

namespace kl::stats {

std::atomic<uint64_t> g_counters[Count];

namespace {

std::string mb(uint64_t b) {
    char s[32];
    snprintf(s, sizeof s, "%.1f MB", double(b) / (1 << 20));
    return s;
}

void run() {
    SetThreadDescription(GetCurrentThread(), L"keylime-status");
    uint64_t last[Count] = {};
    for (;;) {
        Sleep(2000);
        uint64_t now[Count], d[Count];
        uint64_t frame_worst = g_counters[FrameWorstUs].exchange(0), opt_worst = g_counters[OptimizedWorstUs].exchange(0);
        for (int i = 0; i < Count; ++i) now[i] = g_counters[i].load(), d[i] = now[i] - last[i];
        memcpy(last, now, sizeof last);
        PROCESS_MEMORY_COUNTERS pmc{sizeof pmc};
        GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
        KL_INFO("status", "frames %llu (%.1f/s) | GPU: %llu submits, %llu lists, %llu draws, %llu dispatches, %llu copies | "
                "pipelines %llu",
                (unsigned long long)now[Presents], d[Presents] / 2.0, (unsigned long long)d[Submits], (unsigned long long)d[Lists],
                (unsigned long long)d[Draws], (unsigned long long)d[Dispatches], (unsigned long long)d[Copies],
                (unsigned long long)now[Pipelines]);
        KL_INFO("status", "streaming: %llu requests, %s, %llu failed | audio: %llu XMA voices, %llu frames decoded, %llu Opus, %llu cmds | "
                "input readings %llu | RAM %s",
                (unsigned long long)d[StorageRequests], mb(d[StorageBytes]).c_str(), (unsigned long long)now[StorageFailures],
                (unsigned long long)now[XmaVoices], (unsigned long long)d[XmaFrames], (unsigned long long)d[OpusPackets],
                (unsigned long long)d[AudioCommands], (unsigned long long)d[InputReadings], mb(pmc.WorkingSetSize).c_str());
        KL_INFO("status", "host GPU: %llu draws run, %llu skipped, %llu dispatches run | images %llu, %llu uploads, %llu texture writes seen | "
                "frames shown %llu | VRAM: %llu images resident, %llu / %llu MiB, %llu evicted | %llu memory imports",
                (unsigned long long)d[DrawsRun], (unsigned long long)d[DrawsSkipped], (unsigned long long)d[DispatchesRun],
                (unsigned long long)now[Images], (unsigned long long)d[TextureUploads], (unsigned long long)d[TextureWrites],
                (unsigned long long)now[FramesShown], (unsigned long long)now[ImagesResident], (unsigned long long)now[ImageMiB],
                (unsigned long long)now[VramBudgetMiB], (unsigned long long)d[Evictions], (unsigned long long)now[MemoryImports]);
        if (d[FramesShown]) {
            double f = double(d[FramesShown]), ms = 1e-3 / f;
            KL_INFO("status", "frame: %.1f ms (%.1f fps), worst %.1f ms | GPU busy %.1f ms (graphics %.1f, compute %.1f) | CPU recording %.1f ms, "
                    "waiting for pipelines %.1f ms | per frame %.0f draws run, %.0f skipped, %.0f dispatches run, %.0f skipped",
                    2000.0 / f, f / 2.0, frame_worst / 1000.0, (d[GpuUs0] + d[GpuUs1]) * ms, d[GpuUs0] * ms, d[GpuUs1] * ms,
                    d[RecordUs] * ms, d[CompileWaitUs] * ms, d[DrawsRun] / f, d[DrawsSkipped] / f, d[DispatchesRun] / f,
                    d[DispatchesSkipped] / f);
        }
        if (d[ShadersTranslated] || d[ShadersCached] || d[GraphicsBuilt] || d[QuickBuilt] || d[OptimizedBuilt] || now[CompileQueue]) {
            auto each = [](uint64_t us, uint64_t n) { return n ? us / 1000.0 / double(n) : 0.0; };
            KL_INFO("status", "compile: %llu shaders translated (%.1f ms each, %llu KB SPIR-V, %llu%% of blocks still dispatched by number), "
                    "%llu from the disk cache | built %llu graphics (%.0f ms each), %llu quick compute (%.0f ms each), %llu optimized compute "
                    "(%.0f ms each, worst %.1f s) | %llu jobs queued (%llu urgent, %llu early starts, %llu optimized builds) | precompile %llu of %llu",
                    (unsigned long long)d[ShadersTranslated], each(d[TranslateUs], d[ShadersTranslated]), (unsigned long long)(d[SpirvBytes] >> 10),
                    (unsigned long long)(d[BlocksTotal] ? d[BlocksGuarded] * 100 / d[BlocksTotal] : 0), (unsigned long long)d[ShadersCached],
                    (unsigned long long)d[GraphicsBuilt], each(d[GraphicsUs], d[GraphicsBuilt]), (unsigned long long)d[QuickBuilt],
                    each(d[QuickUs], d[QuickBuilt]), (unsigned long long)d[OptimizedBuilt], each(d[OptimizedUs], d[OptimizedBuilt]),
                    opt_worst / 1e6, (unsigned long long)now[CompileQueue], (unsigned long long)now[CompileUrgent],
                    (unsigned long long)now[CompileWarm], (unsigned long long)now[CompileOptimized], (unsigned long long)now[PrecompileDone],
                    (unsigned long long)now[PrecompileListed]);
        }
    }
}

}  // namespace

void start() { std::thread(run).detach(); }

}  // namespace kl::stats
