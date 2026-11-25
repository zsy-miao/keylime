// keylime: named counters every module bumps, and a status thread that
// prints what changed every two seconds (frames, GPU work, streaming,
// audio, input), so the console shows what the title is doing.
#pragma once
#include <atomic>
#include <cstdint>

namespace kl::stats {

enum Counter : int {
    Presents, Submits, Lists, Draws, Dispatches, Copies, Barriers, OtherCommands,
    Pipelines, Resources, StorageRequests, StorageBytes, StorageFailures, FilesOpened,
    XmaVoices, XmaFrames, OpusPackets, AudioCommands, InputReadings, Stubs,
    Images, TextureUploads, TextureWrites, DrawsRun, DispatchesRun, DrawsSkipped, FramesShown, BcpackBytes,
    PipelinesBuilt, WaitingOnCompile, Compiling /* gauge */,
    GpuUs0, GpuUs1, RecordUs, CompileWaitUs, FrameWorstUs /* max */, DispatchesSkipped,
    ShadersTranslated, ShadersCached, TranslateUs, SpirvBytes, BlocksTotal, BlocksGuarded,
    GraphicsBuilt, GraphicsUs, QuickBuilt, QuickUs, OptimizedBuilt, OptimizedUs, OptimizedWorstUs /* max */,
    CompileQueue /* gauge */, PrecompileListed /* gauge */, PrecompileDone,
    // gauges (set, not added): GPU memory state
    ImagesResident, ImageMiB, VramBudgetMiB, MemoryImports, Evictions, Count
};

extern std::atomic<uint64_t> g_counters[Count];
inline void add(Counter c, uint64_t n = 1) { g_counters[c].fetch_add(n, std::memory_order_relaxed); }
inline void set(Counter c, uint64_t v) { g_counters[c].store(v, std::memory_order_relaxed); }
inline void raise(Counter c, uint64_t v) {
    uint64_t now = g_counters[c].load(std::memory_order_relaxed);
    while (v > now && !g_counters[c].compare_exchange_weak(now, v, std::memory_order_relaxed)) {}
}

void start();  // the status thread

}  // namespace kl::stats
