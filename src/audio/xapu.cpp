// keylime: the console's audio decoder (xapu.dll): Opus packets in, float
// PCM out, decoded by libopus. A title connects once for one kind of
// processing, gets memory for its packets and output, queues per-stream
// commands (activate, process, reset, deactivate) and dequeues one result
// per command. Each command is decoded when queued, so its result is ready
// at once. Structures as VIctory found them (apu.rs).
#include <windows.h>

#include <algorithm>
#include <deque>
#include <mutex>
#include <vector>

#include "hle/hle.h"
#include "log.h"
#include "stats.h"
#include "opus.h"

namespace kl::xapu {
namespace {

constexpr HRESULT QUEUE_FULL = HRESULT(0x8AC81000), PENDING_RESULTS = HRESULT(0x8AC81001),
                  INVALID_COMMAND = HRESULT(0x8AC81002), INVALID_STREAM = HRESULT(0x8AC81003),
                  ALREADY_ACTIVE = HRESULT(0x8AC81005), BAD_PACKET = HRESULT(0x8AC81008);
constexpr uint32_t DECODE_OPUS = 1, ACTIVATE = 1, DEACTIVATE = 2, PROCESS = 3, NOTIFY = 4, RESET = 5;
constexpr uint32_t SIGNAL = 1, DEINTERLEAVED_OUTPUT = 4, RESULT_EVENT = 8;
constexpr int MAX_FRAMES = 5760;  // 120 ms at 48 kHz

struct ConnectInput {
    uint32_t process, streams, commands_per_stream, memory, frames_per_output, options;
};
struct ConnectOutput {
    void* memory;
    HANDLE result_event;
};
#pragma pack(push, 1)
struct CommandId {
    uint32_t kind, stream, sequence, options;
    uint64_t context;
};
struct Activate {
    CommandId id;
    uint32_t channels, init_length;
    const void* init;
    float start_pitch;
    uint32_t processing_length;
    void* processing;
};
struct Process {
    CommandId id;
    float pitch_ramp, target_pitch;
    uint32_t first_frame, frame_count, input_length;
    const uint8_t* input;
    uint32_t output_length;
    uint8_t* output;
};
struct Result {
    CommandId id;
    uint32_t state;
    uint64_t state_data, usec;
    HRESULT error;
    uint32_t frames, output_length;
    uint8_t* output;
    float pitch;
    uint32_t input_length;
    const uint8_t* input;
};
#pragma pack(pop)

struct Stream {
    OpusDecoder* dec = nullptr;
    int channels = 0;
};

struct Connection {
    void* memory;
    HANDLE event;
    uint32_t options, per_stream;
    std::mutex lock;
    std::vector<Stream> streams;
    std::vector<uint32_t> queued;
    std::deque<Result> results;
    uint32_t commands = 0, dequeued = 0;
};

HRESULT WINAPI XApuConnect(const ConnectInput* in, ConnectOutput* out, Connection** handle) {
    if (!in || !out || !handle) return E_INVALIDARG;
    if (in->streams < 1 || in->streams > 500 || in->commands_per_stream < 1 || in->commands_per_stream > 16 || !in->memory)
        return E_INVALIDARG;
    if (in->process != DECODE_OPUS) {
        KL_WARN("audio", "audio decoder: processing kind %u not implemented", in->process);
        return E_NOTIMPL;
    }
    if (in->frames_per_output) return E_INVALIDARG;
    auto* c = new Connection();
    c->memory = _aligned_malloc(in->memory, 16);
    memset(c->memory, 0, in->memory);
    c->event = (in->options & RESULT_EVENT) ? CreateEventW(nullptr, FALSE, FALSE, nullptr) : nullptr;
    c->options = in->options;
    c->per_stream = in->commands_per_stream;
    c->streams.resize(in->streams);
    c->queued.assign(in->streams, 0);
    *out = {c->memory, c->event};
    *handle = c;
    KL_INFO("audio", "Opus decoder connected: %u streams, %u bytes", in->streams, in->memory);
    return S_OK;
}

HRESULT WINAPI XApuDisconnect(Connection* c) {
    if (!c) return E_INVALIDARG;
    for (auto& s : c->streams)
        if (s.dec) opus_decoder_destroy(s.dec);
    if (c->event) CloseHandle(c->event);
    _aligned_free(c->memory);
    delete c;
    return S_OK;
}

void process(Stream& s, uint32_t options, const Process& p, Result& r) {
    r.input = p.input, r.input_length = p.input_length, r.output = p.output;
    if (!p.input || !p.output || !p.input_length) {
        r.error = E_INVALIDARG;
        return;
    }
    std::vector<float> pcm(size_t(MAX_FRAMES) * s.channels);
    stats::add(stats::OpusPackets);
    int n = opus_decode_float(s.dec, p.input, int(p.input_length), pcm.data(), MAX_FRAMES, 0);
    if (n < 0) {
        r.error = BAD_PACKET;
        return;
    }
    size_t first = std::min<size_t>(p.first_frame, size_t(n));
    size_t frames = std::min<size_t>({size_t(n) - first, size_t(p.frame_count), p.output_length / (4u * s.channels)});
    auto* out = reinterpret_cast<float*>(p.output);
    const float* kept = pcm.data() + first * s.channels;
    if ((options & DEINTERLEAVED_OUTPUT) && s.channels == 2) {
        for (size_t i = 0; i < frames; ++i) out[i] = kept[2 * i], out[frames + i] = kept[2 * i + 1];
    } else {
        memcpy(out, kept, frames * s.channels * 4);
    }
    r.frames = uint32_t(frames);
    r.output_length = uint32_t(frames * s.channels * 4);
}

HRESULT WINAPI XApuEnqueueCommand(Connection* c, const CommandId* cmd, uint32_t* total) {
    if (!c || !cmd) return E_INVALIDARG;
    CommandId id;
    memcpy(&id, cmd, sizeof id);
    std::lock_guard g(c->lock);
    if (id.stream >= c->streams.size()) return INVALID_STREAM;
    if (c->queued[id.stream] >= c->per_stream) return QUEUE_FULL;
    Result r{};
    r.id = id;
    r.pitch = 1.0f;
    Stream& s = c->streams[id.stream];
    switch (id.kind) {
        case ACTIVATE: {
            Activate a;
            memcpy(&a, cmd, sizeof a);
            if (s.dec) {
                r.error = ALREADY_ACTIVE;
            } else if (a.channels < 1 || a.channels > 2) {
                r.error = E_INVALIDARG;
            } else {
                int err = 0;
                s.dec = opus_decoder_create(48000, int(a.channels), &err);
                s.channels = int(a.channels);
                r.error = err == OPUS_OK ? S_OK : E_INVALIDARG;
            }
            break;
        }
        case DEACTIVATE:
            if (s.dec) opus_decoder_destroy(s.dec), s.dec = nullptr;
            break;
        case RESET:
            if (s.dec) opus_decoder_ctl(s.dec, OPUS_RESET_STATE);
            break;
        case NOTIFY: break;
        case PROCESS: {
            Process p;
            memcpy(&p, cmd, sizeof p);
            if (s.dec) process(s, c->options, p, r);
            else r.error = INVALID_STREAM;
            break;
        }
        default: return INVALID_COMMAND;
    }
    ++c->queued[id.stream];
    ++c->commands;
    c->results.push_back(r);
    if (total) *total = c->commands;
    if ((id.options & SIGNAL) && c->event) SetEvent(c->event);
    return S_OK;
}

HRESULT WINAPI XApuDequeueResult(Connection* c, Result* out) {
    if (!c || !out) return E_INVALIDARG;
    std::lock_guard g(c->lock);
    if (c->results.empty()) return PENDING_RESULTS;
    Result r = c->results.front();
    c->results.pop_front();
    --c->queued[r.id.stream];
    ++c->dequeued;
    memcpy(out, &r, sizeof r);
    return S_OK;
}

uint32_t WINAPI XApuGetResultCount(Connection* c, uint32_t* total) {
    if (!c) return 0;
    std::lock_guard g(c->lock);
    if (total) *total = c->dequeued;
    return uint32_t(c->results.size());
}

}  // namespace

KL_MODULE("xapu", (std::vector<std::string>{"xapu.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(XApuConnect), KL_EXPORT(XApuDisconnect), KL_EXPORT(XApuEnqueueCommand),
                                    KL_EXPORT(XApuDequeueResult), KL_EXPORT(XApuGetResultCount)}));

}  // namespace kl::xapu
