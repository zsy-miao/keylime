// keylime: the console's audio hardware (AcpHal.dll):
//  - the APU heap: host memory with 32-bit device addresses;
//  - SHAPE context arrays allocated from it;
//  - the audio coprocessor's command / message queues (every command
//    completes at once and is acknowledged as the hardware would);
//  - the XMA decoder: enabled XMA contexts are decoded on a thread of ours
//    into their 16-bit PCM output rings, as the hardware does.
// Mixed audio leaves through WASAPI, which the title reaches through COM.
// Design and context word layout as VIctory found them (audio.rs).
#include <windows.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "audio/xma.h"
#include "hle/hle.h"
#include "log.h"
#include "stats.h"

namespace kl::acp {
namespace {

// ---------------------------------------------------------- APU heap ----

constexpr uint32_t APU_BASE = 0x10000;  // 0 means "none" to titles

struct Block {
    size_t size;
    uint32_t apu;
    size_t span;
};
struct Heap {
    std::map<uintptr_t, Block> blocks;   // by host address
    std::map<uint32_t, uintptr_t> by_apu;
    std::map<uint32_t, uint32_t> free;   // start -> end
    bool created = false;
    bool take(uint32_t size, uint32_t align, uint32_t& out) {
        if (free.empty() && !created) free[APU_BASE] = 0xffffffffu;
        created = true;
        for (auto it = free.begin(); it != free.end(); ++it) {
            uint64_t start = it->first, end = it->second;
            uint64_t a = (start + align - 1) / align * align;
            if (a + size > end) continue;
            free.erase(it);
            if (start < a) free[uint32_t(start)] = uint32_t(a);
            if (a + size < end) free[uint32_t(a + size)] = uint32_t(end);
            out = uint32_t(a);
            return true;
        }
        return false;
    }
    void give_back(uint32_t start, uint32_t end) {
        auto it = free.lower_bound(start);
        if (it != free.begin()) {
            auto prev = std::prev(it);
            if (prev->second == start) start = prev->first, free.erase(prev);
        }
        if (auto n = free.find(end); n != free.end()) end = n->second, free.erase(n);
        free[start] = end;
    }
};
std::mutex g_heap_lock;
Heap g_heap;

void* heap_alloc(size_t size, size_t align, uint32_t& apu) {
    if (align < 16) align = 16;
    size_t span = (std::max<size_t>(size, 1) + align - 1) / align * align;
    std::lock_guard g(g_heap_lock);
    if (!g_heap.take(uint32_t(span), uint32_t(align), apu)) return nullptr;
    void* p = _aligned_malloc(span, align);
    if (!p) {
        g_heap.give_back(apu, apu + uint32_t(span));
        return nullptr;
    }
    memset(p, 0, span);
    g_heap.blocks[uintptr_t(p)] = {size, apu, span};
    g_heap.by_apu[apu] = uintptr_t(p);
    return p;
}
void heap_free(void* p) {
    if (!p) return;
    std::lock_guard g(g_heap_lock);
    auto it = g_heap.blocks.find(uintptr_t(p));
    if (it == g_heap.blocks.end()) return;
    g_heap.by_apu.erase(it->second.apu);
    g_heap.give_back(it->second.apu, it->second.apu + uint32_t(it->second.span));
    g_heap.blocks.erase(it);
    _aligned_free(p);
}
uint32_t heap_map(uintptr_t p) {
    std::lock_guard g(g_heap_lock);
    auto it = g_heap.blocks.upper_bound(p);
    if (it == g_heap.blocks.begin()) return 0;
    --it;
    if (p < it->first + std::max<size_t>(it->second.size, 1)) return it->second.apu + uint32_t(p - it->first);
    return 0;
}
// `len` bytes at a device address, if one block holds them all.
uint8_t* heap_bytes(uint32_t apu, size_t len) {
    std::lock_guard g(g_heap_lock);
    auto it = g_heap.by_apu.upper_bound(apu);
    if (it == g_heap.by_apu.begin()) return nullptr;
    --it;
    const Block& b = g_heap.blocks[it->second];
    size_t off = apu - it->first;
    if (off + len > b.size) return nullptr;
    return reinterpret_cast<uint8_t*>(it->second + off);
}

HRESULT WINAPI ApuCreateHeap(uint32_t cached, uint32_t) {
    if (!cached) return E_INVALIDARG;
    std::lock_guard g(g_heap_lock);
    if (g_heap.created) return HRESULT(0x8ACA0001);  // already allocated
    g_heap.created = true;
    g_heap.free[APU_BASE] = 0xffffffffu;
    KL_INFO("audio", "APU heap: %u bytes cached", cached);
    return S_OK;
}
void* WINAPI ApuMapApuAddress(uint32_t apu) { return heap_bytes(apu, 1); }
HRESULT WINAPI ApuAlloc(void** out, uint32_t* apu, uint32_t size, uint32_t align, uint32_t) {
    if (!out || !size || align < 4 || (align & (align - 1))) return E_INVALIDARG;
    uint32_t a;
    void* p = heap_alloc(size, align, a);
    if (!p) return E_OUTOFMEMORY;
    *out = p;
    if (apu) *apu = a;
    return S_OK;
}
void WINAPI ApuFree(void* p) { heap_free(p); }
uint32_t WINAPI ApuMapVirtualAddress(const void* p) { return heap_map(uintptr_t(p)); }
BOOL WINAPI ApuIsVirtualAddressValid(const void* p, uint32_t align) {
    uint32_t a = heap_map(uintptr_t(p));
    return a && (!align || a % align == 0);
}

// ----------------------------------------------------- XMA contexts ----
// Context words (16 per context in the shared array):
// 0: input0 packets (0-11), loop count (12-19, 255 forever), valid inputs (20-21),
//    output blocks (22-26), write block (27-31, hardware)
// 1: input1 packets (0-11), loop end subframe (12-13), loop-start subframes to skip (17-19),
//    start subframes to skip (24-26), rate (27-28: 24/32/44.1/48 kHz), stereo (29), writable (31)
// 2: read position in bits (0-25), error status (26-30), error (31)
// 3: loop start bits; 4: loop end bits (0-25), current input (31); 5,6: inputs; 7: output; 9: read block (0-4)

constexpr uint32_t ENABLE_XMA = 5, ENABLE_XMAS = 6, DISABLE_XMA = 7, DISABLE_XMAS = 8, ADVANCE_XMA_READ = 18;
constexpr size_t BLOCK_BYTES = 256, SAMPLES_PER_BLOCK = BLOCK_BYTES / 2;
constexpr uint32_t INPUT_INVALID = 4;

struct Context {
    volatile uint32_t* w;
    uint32_t get(int i) const { return w[i]; }
    void set(int i, uint32_t v) { w[i] = v; }
    uint32_t field(int i, int shift, int bits) const { return (w[i] >> shift) & ((1u << bits) - 1); }
    void set_field(int i, int shift, int bits, uint32_t v) {
        uint32_t mask = ((1u << bits) - 1) << shift;
        w[i] = (w[i] & ~mask) | ((v << shift) & mask);
    }
    size_t channels() const { return 1 + field(1, 29, 1); }
};

struct Voice {
    std::unique_ptr<xma::Decoder> decoder;
    std::deque<int16_t> pcm;
    size_t skip = 0;
    bool ended = false;
};

struct Xma {
    uintptr_t array = 0;
    std::vector<std::unique_ptr<std::mutex>> locks;
    std::vector<std::unique_ptr<Voice>> voices;  // null while disabled
    bool context(uint32_t i, Context& c) const {
        if (i >= voices.size() || !array) return false;
        c.w = reinterpret_cast<volatile uint32_t*>(array + size_t(i) * 64);
        return true;
    }
};
std::mutex g_xma_setup;
Xma g_xma;
std::atomic<uint64_t> g_frames{0}, g_bad{0}, g_enabled{0};

void wake();

void enable(uint32_t i) {
    Context c;
    if (!g_xma.context(i, c)) return;
    std::lock_guard g(*g_xma.locks[i]);
    c.set_field(2, 26, 6, 0);
    auto v = std::make_unique<Voice>();
    v->skip = c.field(1, 24, 3) * xma::SUBFRAME_SAMPLES;
    g_xma.voices[i] = std::move(v);
    stats::add(stats::XmaVoices);
    uint64_t n = ++g_enabled;
    if (n == 1 || (n & (n - 1)) == 0) KL_INFO("audio", "%llu XMA voices enabled so far", (unsigned long long)n);
    wake();
}
void disable(uint32_t i) {
    Context c;
    if (!g_xma.context(i, c)) return;
    std::lock_guard g(*g_xma.locks[i]);
    if (g_xma.voices[i]) stats::g_counters[stats::XmaVoices].fetch_sub(1);
    g_xma.voices[i].reset();
}
void advance_read(uint32_t i, uint32_t blocks) {
    Context c;
    if (!g_xma.context(i, c)) return;
    {
        std::lock_guard g(*g_xma.locks[i]);
        uint32_t size = std::max(1u, c.field(0, 22, 5));
        uint32_t read = (c.field(9, 0, 5) + blocks % size) % size;
        c.set_field(9, 0, 5, read);
        if (blocks) c.set_field(1, 31, 1, 1);
    }
    wake();
}

// Decodes the next frame of the current input into the voice's PCM; false when nothing is left.
bool decode(Context c, Voice& v) {
    if (v.ended) return false;
    size_t channels = c.channels();
    uint32_t current = c.field(4, 31, 1), valid = c.field(0, 20, 2);
    if (!(valid & (1u << current))) {
        if (!(valid & (1u << (1 - current)))) {
            if (v.decoder) {
                std::vector<int16_t> tail;
                v.decoder->flush(tail);
                v.pcm.insert(v.pcm.end(), tail.begin(), tail.end());
            }
            v.ended = true;
            c.set_field(2, 26, 5, INPUT_INVALID);
            c.set_field(2, 31, 1, 1);
            return !v.pcm.empty();
        }
        current = 1 - current;
        c.set_field(4, 31, 1, current);
        c.set_field(2, 0, 26, 0);
    }
    uint32_t address = current ? c.get(6) : c.get(5);
    uint32_t packets = current ? c.field(1, 0, 12) : c.field(0, 0, 12);
    size_t len = size_t(packets) * xma::PACKET_BYTES;
    const uint8_t* data = heap_bytes(address & ~0x7ffu, len);
    if (!data) {
        c.set_field(0, 20 + current, 1, 0);
        return true;
    }
    size_t at = c.field(2, 0, 26);
    if (at < 32) {
        auto p = xma::packet(data, len, 0);
        at = 32 + (p ? p->first_frame_bit : 0);
    }
    auto frame = xma::read_frame(data, len, at);
    if (!frame) {
        c.set_field(0, 20 + current, 1, 0);
        return true;
    }
    if (!v.decoder) {
        static const uint32_t rates[4] = {24000, 32000, 44100, 48000};
        v.decoder = std::make_unique<xma::Decoder>(channels, rates[c.field(1, 27, 2)]);
    }
    std::vector<int16_t> pcm;
    pcm.reserve(xma::FRAME_SAMPLES * channels);
    ++g_frames;
    stats::add(stats::XmaFrames);
    if (!v.decoder->decode(frame->data.data(), frame->bits, pcm)) ++g_bad;
    size_t next = frame->next.value_or(0);
    uint32_t loops = c.field(0, 12, 8);
    size_t loop_end = c.field(4, 0, 26);
    bool ends_loop = loop_end && (frame->start == loop_end || (frame->start < loop_end && (!frame->next || loop_end < *frame->next)));
    if (loops > 0 && ends_loop) {
        size_t keep = (size_t(c.field(1, 12, 2)) + 1) * xma::SUBFRAME_SAMPLES * channels;
        if (pcm.size() > keep) pcm.resize(keep);
        next = c.field(3, 0, 26);
        v.skip = c.field(1, 17, 3) * xma::SUBFRAME_SAMPLES;
        if (loops != 255) c.set_field(0, 12, 8, loops - 1);
    }
    size_t skip = std::min(v.skip * channels, pcm.size());
    v.skip -= skip / channels;
    v.pcm.insert(v.pcm.end(), pcm.begin() + skip, pcm.end());
    if (!next) c.set_field(0, 20 + current, 1, 0);
    c.set_field(2, 0, 26, uint32_t(next));
    return true;
}

void pump(Context c, Voice& v) {
    uint32_t size = c.field(0, 22, 5);
    if (!size) return;
    while (c.field(1, 31, 1) == 1) {
        if (v.pcm.size() < SAMPLES_PER_BLOCK && !decode(c, v)) return;
        if (v.pcm.size() < SAMPLES_PER_BLOCK && !v.ended) continue;
        uint32_t write = c.field(0, 27, 5);
        uint8_t* out = heap_bytes(c.get(7) + write * uint32_t(BLOCK_BYTES), BLOCK_BYTES);
        if (!out) return;
        auto* s = reinterpret_cast<int16_t*>(out);
        for (size_t k = 0; k < SAMPLES_PER_BLOCK; ++k) {
            if (v.pcm.empty()) s[k] = 0;
            else s[k] = v.pcm.front(), v.pcm.pop_front();
        }
        write = (write + 1) % size;
        c.set_field(0, 27, 5, write);
        if (write == c.field(9, 0, 5)) c.set_field(1, 31, 1, 0);  // full until the title reads
    }
}

std::mutex g_wake_lock;
std::condition_variable g_wake_cv;
bool g_woken = false;

void decoder_thread() {
    SetThreadDescription(GetCurrentThread(), L"keylime-xma");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    for (;;) {
        {
            std::unique_lock g(g_wake_lock);
            // A 256-byte block lasts 2.7 ms at 48 kHz mono.
            g_wake_cv.wait_for(g, std::chrono::milliseconds(2), [] { return g_woken; });
            g_woken = false;
        }
        std::lock_guard setup(g_xma_setup);  // contexts may be released and reallocated meanwhile
        for (uint32_t i = 0; i < g_xma.voices.size(); ++i) {
            std::lock_guard g(*g_xma.locks[i]);
            Context c;
            if (g_xma.voices[i] && g_xma.context(i, c)) pump(c, *g_xma.voices[i]);
        }
    }
}

void wake() {
    static std::once_flag once;
    std::call_once(once, [] { std::thread(decoder_thread).detach(); });
    {
        std::lock_guard g(g_wake_lock);
        g_woken = true;
    }
    g_wake_cv.notify_one();
}

bool xma_command(uint32_t command, const uint32_t* data) {
    auto word = [&](size_t i) -> uint32_t { return data ? data[i] : 0; };
    switch (command) {
        case ENABLE_XMA: enable(data ? word(0) : ~0u); return true;
        case DISABLE_XMA: disable(data ? word(0) : ~0u); return true;
        case ENABLE_XMAS:
        case DISABLE_XMAS:
            for (uint32_t i = 0; i < g_xma.voices.size(); ++i)
                if (word(i / 32) & (1u << (i % 32))) command == ENABLE_XMAS ? enable(i) : disable(i);
            return true;
        case ADVANCE_XMA_READ: advance_read(data ? word(0) : ~0u, word(1)); return true;
        default: return false;
    }
}

// --------------------------------------------------- SHAPE contexts ----

#pragma pack(push, 1)
struct ShapeContexts {
    uint32_t counts[6];  // rate conversion, EQ/compression, filter/volume, DMA, XMA, PCM
    void* arrays[6];
    uint32_t apu[6];
};
#pragma pack(pop)
const size_t CONTEXT_BYTES[6] = {32, 256, 256, 16, 64, 256};
bool g_contexts = false;
void* g_arrays[6] = {};

HRESULT WINAPI AcpHalAllocateShapeContexts(ShapeContexts* s) {
    if (!s) return E_INVALIDARG;
    std::lock_guard g(g_xma_setup);
    if (g_contexts) return E_FAIL;
    uint32_t counts[6];
    memcpy(counts, s->counts, sizeof counts);
    void* arrays[6] = {};
    uint32_t apu[6] = {};
    for (int i = 0; i < 6; ++i) {
        if (!counts[i]) continue;
        arrays[i] = heap_alloc(counts[i] * CONTEXT_BYTES[i], 256, apu[i]);
        if (!arrays[i]) {
            for (auto* p : arrays) heap_free(p);
            return E_OUTOFMEMORY;
        }
    }
    memcpy(s->arrays, arrays, sizeof arrays);
    memcpy(s->apu, apu, sizeof apu);
    memcpy(g_arrays, arrays, sizeof arrays);
    g_contexts = true;
    g_xma.array = uintptr_t(arrays[4]);
    g_xma.locks.clear();
    g_xma.voices.clear();
    for (uint32_t i = 0; i < counts[4]; ++i) {
        g_xma.locks.push_back(std::make_unique<std::mutex>());
        g_xma.voices.push_back(nullptr);
    }
    KL_INFO("audio", "SHAPE contexts: SRC %u, EQ %u, FLT %u, DMA %u, XMA %u, PCM %u", counts[0], counts[1], counts[2],
            counts[3], counts[4], counts[5]);
    return S_OK;
}
HRESULT WINAPI AcpHalReleaseShapeContexts() {
    std::lock_guard g(g_xma_setup);
    if (!g_contexts) return E_FAIL;
    g_xma = Xma();
    for (auto*& p : g_arrays) heap_free(p), p = nullptr;
    g_contexts = false;
    return S_OK;
}

// --------------------------------------------------- the coprocessor ----

constexpr uint32_t LOAD_FLOWGRAPH = 1, REGISTER_MESSAGES = 2, UNREGISTER_MESSAGES = 3;
constexpr uint32_t FLOWGRAPH_COMPLETED = 0x02, COMMAND_COMPLETED = 0x10, DISCONNECTED = 0x80;
constexpr uint64_t FRAMES_PER_SECOND = 375;  // 128-sample frames at 48 kHz

#pragma pack(push, 1)
struct Message {
    uint32_t kind, dropped, usec;
    uint8_t data[16];
};
#pragma pack(pop)

struct Acp {
    const void* const* vtable;
    std::mutex lock;
    bool connected = false;
    uint32_t mask = 0;
    std::deque<Message> messages;
    size_t capacity = 0;
    uint32_t dropped = 0;
    std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
    uint64_t usec() const {
        return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - since).count());
    }
    void post(uint32_t kind, const void* data, size_t len) {
        if (!(mask & kind)) return;
        if (messages.size() >= capacity) {
            ++dropped;
            return;
        }
        Message m{kind, dropped, uint32_t(usec()), {}};
        dropped = 0;
        memcpy(m.data, data, std::min<size_t>(len, 16));
        messages.push_back(m);
    }
};

HRESULT WINAPI connect(Acp* a, uint32_t commands, uint32_t messages) {
    if (!commands) return E_INVALIDARG;
    std::lock_guard g(a->lock);
    a->connected = true;
    a->capacity = messages;
    KL_INFO("audio", "audio coprocessor connected: %u commands, %u messages", commands, messages);
    return S_OK;
}
HRESULT WINAPI disconnect(Acp* a) {
    std::lock_guard g(a->lock);
    if (!a->connected) return E_FAIL;
    a->post(DISCONNECTED, nullptr, 0);
    a->connected = false;
    return S_OK;
}
HRESULT WINAPI submit(Acp* a, uint32_t command, uint64_t id, uint32_t, const uint32_t* data, uint32_t notification) {
    stats::add(stats::AudioCommands);
    static std::atomic<uint64_t> seen{0};
    if (command < 64 && !(seen.fetch_or(1ull << command) & (1ull << command)))
        KL_INFO("audio", "first coprocessor command %u", command);
    if (!xma_command(command, data)) {
        std::lock_guard g(a->lock);
        if (command == REGISTER_MESSAGES) a->mask |= data ? data[0] : 0;
        else if (command == UNREGISTER_MESSAGES) a->mask &= ~(data ? data[0] : 0);
        else if (command == LOAD_FLOWGRAPH) {
            uint32_t fg = data ? data[1] : 0;
            a->post(FLOWGRAPH_COMPLETED, &fg, 4);
        }
        // Other SHAPE units' commands are acknowledged but not carried out yet.
    }
    std::lock_guard g(a->lock);
    if (id) {
        uint8_t done[16];
        uint32_t frame = uint32_t(a->usec() * FRAMES_PER_SECOND / 1000000);
        memcpy(done, &command, 4);
        memcpy(done + 4, &id, 8);
        memcpy(done + 12, &frame, 4);
        a->post(COMMAND_COMPLETED, done, 16);
    }
    if (notification)
        if (auto* w = heap_bytes(notification, 4)) *reinterpret_cast<volatile uint32_t*>(w) = 1;
    return S_OK;
}
bool WINAPI pop_message(Acp* a, Message* out) {
    std::lock_guard g(a->lock);
    if (a->messages.empty()) return false;
    if (out) *out = a->messages.front();
    a->messages.pop_front();
    return true;
}
uint32_t WINAPI message_count(Acp* a) {
    std::lock_guard g(a->lock);
    return uint32_t(a->messages.size());
}
void WINAPI release(Acp* a) { delete a; }

const void* const kAcp[6] = {(void*)connect, (void*)disconnect, (void*)submit, (void*)pop_message, (void*)message_count, (void*)release};

HRESULT WINAPI AcpHalCreate(Acp** out) {
    if (!out) return E_INVALIDARG;
    auto* a = new Acp();
    a->vtable = kAcp;
    *out = a;
    return S_OK;
}

}  // namespace

KL_MODULE("acphal", (std::vector<std::string>{"acphal.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(AcpHalAllocateShapeContexts), KL_EXPORT(AcpHalReleaseShapeContexts),
                                    KL_EXPORT(AcpHalCreate), KL_EXPORT(ApuAlloc), KL_EXPORT(ApuCreateHeap), KL_EXPORT(ApuFree),
                                    KL_EXPORT(ApuIsVirtualAddressValid), KL_EXPORT(ApuMapApuAddress),
                                    KL_EXPORT(ApuMapVirtualAddress)}));

}  // namespace kl::acp
