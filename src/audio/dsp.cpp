// keylime: the console's audio DSP (chad.dll): streams that convolve blocks
// with an impulse response (reverb) or run other block transforms. Commands
// complete when submitted; outputs are silence for now (the hardware takes
// impulse responses in its own partitioned frequency-domain layout, not yet
// known), so reverb returns are dry. Shapes as VIctory found them (dsp.rs).
#include <windows.h>

#include <atomic>
#include <mutex>

#include "hle/hle.h"
#include "log.h"

namespace kl::dsp {
namespace {

constexpr uint32_t CONVOLUTION = 1, STEREO_OPTIONS = 2 | 8, NOT_FOR_CONVOLUTION = 4;

struct Dsp {
    uint32_t max_streams;
    std::mutex lock;
    uint32_t active = 0;
};
#pragma pack(push, 1)
struct StreamDesc {
    uint32_t kind, options, frames, channels, ir_floats;
    const float* ir;
};
#pragma pack(pop)
struct Status {
    volatile uint32_t submitted, completed, reserved;
    volatile HRESULT result;
};
struct Stream {
    Dsp* dsp;
    uint32_t frames, channels;
    Status status;
};
struct Command {
    float gains[2];
    const float* input;
    float* output;
    uint64_t reserved;
};

HRESULT WINAPI XDspConnectWithMaximumStreamLimit(void* base, uint32_t len, uint32_t ir_seconds, uint32_t max_streams,
                                                 uint32_t max_channels, Dsp** out) {
    if (!base || len < 0x10000 || len > 0x1F400000 || ir_seconds > 0x80 || max_streams > 0x100 || max_channels > 2)
        return E_INVALIDARG;
    if (!out) return E_POINTER;
    auto* d = new Dsp();
    d->max_streams = max_streams;
    *out = d;
    KL_INFO("audio", "audio DSP connected: %u streams, %u s of impulse response", max_streams, ir_seconds);
    return S_OK;
}
HRESULT WINAPI XDspDisconnect(Dsp* d) {
    if (!d) return E_INVALIDARG;
    delete d;
    return S_OK;
}
bool valid(const StreamDesc& d) {
    bool shape = d.kind == CONVOLUTION
                     ? (d.channels >= 1 && d.channels <= 2 && (d.frames == 0x200 || d.frames == 0x400) && !(d.options & NOT_FOR_CONVOLUTION))
                     : (d.channels == 1 && (d.frames == 0x200 || d.frames == 0x400 || d.frames == 0x800));
    return shape && (!(d.options & STEREO_OPTIONS) || d.channels == 2);
}
HRESULT WINAPI XDspActivate(Dsp* d, const StreamDesc* desc, const Status** status, Stream** stream) {
    if (!status || !stream) return E_POINTER;
    if (!d || !desc || !valid(*desc)) return E_INVALIDARG;
    {
        std::lock_guard g(d->lock);
        if (d->active >= d->max_streams) return E_OUTOFMEMORY;
        ++d->active;
    }
    auto* s = new Stream{d, desc->frames, desc->channels, {}};
    *status = &s->status;
    *stream = s;
    return S_OK;
}
HRESULT WINAPI XDspDeactivate(Stream* s) {
    if (!s) return E_INVALIDARG;
    {
        std::lock_guard g(s->dsp->lock);
        --s->dsp->active;
    }
    delete s;
    return S_OK;
}
HRESULT WINAPI XDspSubmitCommand(Stream* s, const Command* c, uint32_t* ticket) {
    if (!s || !c) return E_INVALIDARG;
    if (c->output) memset(c->output, 0, size_t(s->frames) * s->channels * sizeof(float));
    uint32_t next = s->status.submitted + 1;
    s->status.submitted = next;
    s->status.result = S_OK;
    s->status.completed = next;
    if (ticket) *ticket = next;
    return S_OK;
}

}  // namespace

KL_MODULE("chad", (std::vector<std::string>{"chad.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(XDspConnectWithMaximumStreamLimit), KL_EXPORT(XDspDisconnect), KL_EXPORT(XDspActivate),
                                    KL_EXPORT(XDspDeactivate), KL_EXPORT(XDspSubmitCommand)}));

}  // namespace kl::dsp
