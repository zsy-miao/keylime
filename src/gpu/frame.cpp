// keylime GPU: frame pacing. The title sets a frame interval and waits for
// each frame event (origin, vblank, flip). Ours is a clock started by the
// first interval, ticking once per interval; tokens count frames. Each event
// kind happens once per tick, so two waits for the same kind never end on the
// same tick (the title polls the origin with "token on time-out" and is paced
// by a blocking flip wait). Behaviour after VIctory's frame.rs; waits use
// high-resolution waitable timers instead of Sleep (whose ~1-15 ms overshoot
// shows up as stutter).
#include <windows.h>

#include <atomic>
#include <chrono>
#include <mutex>

#include "gpu/device.h"
#include "log.h"

namespace kl::gpu::frame {
namespace {

using clk = std::chrono::steady_clock;
constexpr HRESULT TIMEOUT = S_FALSE;
constexpr uint32_t ORIGIN = 0, TOKEN_ON_TIMEOUT = 2;

std::mutex g_lock;
bool g_started = false;
clk::time_point g_start;
std::chrono::nanoseconds g_interval{16'666'667};
uint64_t g_first = 0;
std::atomic<uint64_t> g_frame{0};
std::atomic<uint64_t> g_last[3];

std::atomic<uint64_t>& last(uint32_t kind) { return g_last[kind < 3 ? kind : 2]; }

// Sleeps precisely (high-resolution waitable timer per thread).
void sleep_for(std::chrono::nanoseconds d) {
    if (d.count() <= 0) return;
    thread_local HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    if (!timer) {
        Sleep(DWORD(d.count() / 1'000'000));
        return;
    }
    LARGE_INTEGER due;
    due.QuadPart = -(d.count() / 100);  // relative, 100 ns units
    if (!due.QuadPart) due.QuadPart = -1;
    SetWaitableTimerEx(timer, &due, 0, nullptr, nullptr, nullptr, 0);
    WaitForSingleObject(timer, INFINITE);
}

void next_tick(uint32_t kind, uint64_t& tick, std::chrono::nanoseconds& wait) {
    std::lock_guard g(g_lock);
    if (!g_started) g_started = true, g_start = clk::now();
    auto elapsed = clk::now() - g_start;
    int64_t iv = g_interval.count() > 0 ? g_interval.count() : 1;
    tick = g_first + uint64_t(elapsed.count() / iv) + 1;
    if (tick < last(kind).load() + 1) tick = last(kind).load() + 1;
    auto at = std::chrono::nanoseconds(int64_t(tick - g_first) * iv);
    wait = at - std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
}

HRESULT WINAPI set_interval(void*, void*, uint32_t length_us, uint32_t period, uint32_t) {
    std::lock_guard g(g_lock);
    g_interval = std::chrono::microseconds(length_us ? length_us : 16667);
    uint64_t first = 0;
    for (auto& l : g_last) first = l.load() > first ? l.load() : first;
    g_first = first;
    g_start = clk::now();
    g_started = true;
    KL_INFO("gpu", "frame interval %u us (period %u)", length_us, period);
    return S_OK;
}

HRESULT WINAPI schedule(void*, uint32_t, uint32_t, const void*, uint32_t) { return S_OK; }

HRESULT WINAPI wait(void*, uint32_t kind, uint32_t timeout_ms, const void*, uint32_t flags, uint64_t* token) {
    uint64_t tick;
    std::chrono::nanoseconds w;
    next_tick(kind, tick, w);
    auto limit = std::chrono::milliseconds(timeout_ms);
    HRESULT hr;
    if (w > limit) {
        sleep_for(limit);
        hr = TIMEOUT;
    } else {
        sleep_for(w);
        uint64_t prev = last(kind).load();
        while (prev < tick && !last(kind).compare_exchange_weak(prev, tick)) {}
        hr = S_OK;
    }
    if (kind == ORIGIN && (hr == S_OK || (flags & TOKEN_ON_TIMEOUT))) {
        uint64_t f = ++g_frame;
        if (token) *token = f;
    }
    return hr;
}

HRESULT WINAPI statistics(void*, uint64_t, uint32_t, uint32_t* count, void*) {
    if (count) *count = 0;
    return S_OK;
}

struct Init {
    Init() {
        device_set(78, (void*)&set_interval);  // 0x270
        device_set(79, (void*)&schedule);      // 0x278
        device_set(80, (void*)&wait);          // 0x280
        device_set(81, (void*)&statistics);    // 0x288
    }
} g_init;

}  // namespace

uint64_t current() { return g_frame.load(); }

}  // namespace kl::gpu::frame
