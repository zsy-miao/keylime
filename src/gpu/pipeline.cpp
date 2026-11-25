// keylime GPU: pipeline state objects. For now they keep copies of their
// shader binaries (RDNA 2 code) and the stream they came from; translation
// happens later, first for the UI shaders.
#include <mutex>
#include <vector>

#include "gpu/device.h"
#include "gpu/fence.h"
#include "log.h"
#include "stats.h"

namespace kl::gpu::pipeline {
namespace {

const Iid PIPELINE_IID = guid(0x765a30f3, 0xf624, 0x4c6f, {0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45});

struct Bytecode {
    const void* code;
    size_t size;
};

struct Shader {
    uint32_t stage;  // 0 VS 1 PS 2 DS 3 HS 4 GS 5 CS 6 AS 7 MS
    std::vector<uint8_t> code;
};

struct Pipeline {
    Object object;
    bool compute;
    void* root;
    std::vector<Shader> shaders;
};

std::atomic<uint64_t> g_count{0}, g_bytes{0};

HRESULT WINAPI query(Pipeline* p, const Iid* iid, void** out) {
    bool ok = same(iid, PIPELINE_IID);
    for (auto& g : fence::PAGEABLE_IIDS) ok |= same(iid, g);
    return answer(Kind::Pipeline, iid, ok ? p : nullptr, out);
}
HRESULT WINAPI cached_blob(Pipeline*, void** out) {
    if (out) *out = nullptr;
    return E_NOTIMPL;
}
void** table() {
    static void** t = [] {
        void** t = make_table(Kind::Pipeline, 24);
        fill_object(t, (void*)&query);
        fill_child(t);
        t[8] = (void*)&cached_blob;  // 0x40 GetCachedBlob
        return t;
    }();
    return t;
}

void keep(Pipeline* p, uint32_t stage, const Bytecode& b) {
    if (!b.code || !b.size) return;
    auto* c = static_cast<const uint8_t*>(b.code);
    p->shaders.push_back({stage, std::vector<uint8_t>(c, c + b.size)});
    g_bytes += b.size;
}

HRESULT finish(Pipeline* p, const Iid* iid, void** out) {
    stats::add(stats::Pipelines);
    uint64_t n = ++g_count;
    if ((n & (n - 1)) == 0) KL_INFO("gpu", "%llu pipelines created (%llu KB of shader code)", (unsigned long long)n, (unsigned long long)(g_bytes >> 10));
    return query(p, iid, out);
}

// PC layouts: root signature, then VS, PS, DS, HS, GS bytecode (graphics) or CS (compute).
struct GraphicsHead {
    void* root;
    Bytecode stages[5];
};
struct ComputeHead {
    void* root;
    Bytecode cs;
};

HRESULT WINAPI create_graphics(void*, const GraphicsHead* d, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    auto* p = new Pipeline{{table(), 0, {0}}, false, d->root, {}};
    for (uint32_t i = 0; i < 5; ++i) keep(p, i, d->stages[i]);
    return finish(p, iid, out);
}
HRESULT WINAPI create_compute(void*, const ComputeHead* d, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    auto* p = new Pipeline{{table(), 0, {0}}, true, d->root, {}};
    keep(p, 5, d->cs);
    return finish(p, iid, out);
}
HRESULT WINAPI create_compute_x(void* dev, const ComputeHead* d, uint32_t, const void*, const Iid* iid, void** out) {
    return create_compute(dev, d, iid, out);
}

// A pipeline as a stream of subobjects: (type u32, data) each padded to 8 bytes.
struct StreamDesc {
    size_t size;
    const uint8_t* stream;
};
// Subobject sizes (type included) and shader stages; the console numbers amplification and mesh
// shaders 23 and 24 (PC: 24, 25). Table from VIctory (pipeline.rs).
bool subobject(uint32_t kind, size_t& size, int& stage) {
    stage = -1;
    switch (kind) {
        case 1: case 2: case 3: case 4: case 5: case 6: stage = int(kind) - 1; size = 24; return true;
        case 23: stage = 6; size = 24; return true;
        case 24: stage = 7; size = 24; return true;
        case 0: case 17: size = 16; return true;
        case 12: case 19: size = 24; return true;
        case 7: case 15: size = 40; return true;
        case 8: size = 336; return true;
        case 10: size = 48; return true;
        case 11: size = 56; return true;
        case 21: size = 64; return true;
        case 22: size = 32; return true;
        case 9: case 13: case 14: case 16: case 18: case 20: size = 8; return true;
        default: return false;
    }
}

HRESULT WINAPI create_stream(void*, const StreamDesc* d, const Iid* iid, void** out) {
    if (!d || !d->stream) return E_INVALIDARG;
    auto* p = new Pipeline{{table(), 0, {0}}, false, nullptr, {}};
    for (size_t at = 0; at + 8 <= d->size;) {
        uint32_t kind;
        memcpy(&kind, d->stream + at, 4);
        size_t size;
        int stage;
        if (!subobject(kind, size, stage)) {
            KL_WARN("gpu", "pipeline subobject %u of unknown size at %zu; the rest is ignored", kind, at);
            break;
        }
        if (at + size > d->size) break;
        if (kind == 0) memcpy(&p->root, d->stream + at + 8, 8);
        if (stage >= 0) {
            Bytecode b;
            memcpy(&b, d->stream + at + 8, sizeof b);
            keep(p, uint32_t(stage), b);
            if (stage == 5) p->compute = true;
        }
        at += size;
    }
    return finish(p, iid, out);
}

struct Init {
    Init() {
        device_set(10, (void*)&create_graphics);   // 0x50
        device_set(11, (void*)&create_compute);    // 0x58
        device_set(62, (void*)&create_compute_x);  // 0x1f0
        device_set(86, (void*)&create_stream);     // 0x2b0
    }
} g_init;

}  // namespace
}  // namespace kl::gpu::pipeline
