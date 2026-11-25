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
    }
}

}  // namespace

void start() { std::thread(run).detach(); }

}  // namespace kl::stats
