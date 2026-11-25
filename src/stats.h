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
    Images, TextureUploads, TextureWrites, DrawsRun, DispatchesRun, DrawsSkipped, FramesShown, Count
};

extern std::atomic<uint64_t> g_counters[Count];
inline void add(Counter c, uint64_t n = 1) { g_counters[c].fetch_add(n, std::memory_order_relaxed); }

void start();  // the status thread

}  // namespace kl::stats
