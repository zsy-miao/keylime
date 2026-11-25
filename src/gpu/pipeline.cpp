// keylime GPU: pipeline state objects. They keep copies of their shader
// binaries (RDNA 2 code) and fixed-function state; the executor starts
// translating and compiling them as soon as they are created.
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <vector>

#include "gpu/pipeline.h"

#include "gpu/device.h"
#include "gpu/exec.h"
#include "gpu/fence.h"
#include "gpu/texture.h"
#include "log.h"
#include "stats.h"

namespace kl::gpu::pipeline {
namespace {

const Iid PIPELINE_IID = guid(0x765a30f3, 0xf624, 0x4c6f, {0xa8, 0x28, 0xac, 0xe9, 0x48, 0x62, 0x24, 0x45});

struct Bytecode {
    const void* code;
    size_t size;
};

struct Pipeline {
    Object object;
    State state;
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

// KEYLIME_DUMP_SHADERS=<dir>: writes the first 400 distinct shader blobs there (for study).
void dump(uint32_t stage, const uint8_t* c, size_t n) {
    static char dir[MAX_PATH] = {};
    static bool on = GetEnvironmentVariableA("KEYLIME_DUMP_SHADERS", dir, MAX_PATH) > 0;
    static std::atomic<int> count{0};
    if (!on || count >= 400) return;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; ++i) h = (h ^ c[i]) * 0x100000001b3ull;
    char path[MAX_PATH + 64];
    static const char* names[] = {"vs", "ps", "ds", "hs", "gs", "cs", "as", "ms"};
    snprintf(path, sizeof path, "%s/%s-%016llx.bin", dir, names[stage & 7], (unsigned long long)h);
    if (GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES) return;
    if (FILE* f = fopen(path, "wb")) {
        fwrite(c, 1, n, f);
        fclose(f);
        ++count;
    }
}

void keep(Pipeline* p, uint32_t stage, const Bytecode& b) {
    if (!b.code || !b.size) return;
    auto* c = static_cast<const uint8_t*>(b.code);
    dump(stage, c, b.size);
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < b.size; ++i) h = (h ^ c[i]) * 0x100000001b3ull;
    p->state.shaders.push_back({stage, std::vector<uint8_t>(c, c + b.size), h});
    g_bytes += b.size;
}

// D3D12 defaults for subobjects a stream leaves out.
void defaults(State& s) {
    for (auto& b : s.blend) b = {0, 0, 2, 1, 1, 2, 1, 1, 4, 0xf};
    s.raster = {3, 3, 0, 0, 0.0f, 0.0f, 1, 0, 0, 0, 0};
    s.depth = {1, 1, 2, 0, 0xff, 0xff, {1, 1, 1, 8}, {1, 1, 1, 8}, 0};
}
// Fixed-function pieces, as laid out in D3D12 descriptions.
void read_blend(State& s, const uint8_t* p) {
    memcpy(&s.alpha_to_coverage, p, 4);
    memcpy(&s.independent_blend, p + 4, 4);
    for (int i = 0; i < 8; ++i) {
        uint32_t w[10];
        memcpy(w, p + 8 + 40 * i, 40);
        s.blend[i] = {w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9] & 0xff};
    }
    if (!s.independent_blend)
        for (int i = 1; i < 8; ++i) s.blend[i] = s.blend[0];
}
void read_raster(State& s, const uint8_t* p) { memcpy(&s.raster, p, 44); }
void read_depth(State& s, const uint8_t* p, bool bounds) {
    uint32_t w[4];
    memcpy(w, p, 16);
    s.depth.depth_enable = w[0], s.depth.depth_write = w[1], s.depth.depth_func = w[2], s.depth.stencil_enable = w[3];
    s.depth.read_mask = p[16], s.depth.write_mask = p[17];
    memcpy(&s.depth.front, p + 20, 16);
    memcpy(&s.depth.back, p + 36, 16);
    if (bounds) memcpy(&s.depth.bounds, p + 52, 4);
}
void read_input(State& s, const uint8_t* p) {
    const uint8_t* elems;
    uint32_t n;
    memcpy(&elems, p, 8);
    memcpy(&n, p + 8, 4);
    if (!elems) return;
    uint32_t append = 0;
    std::vector<uint32_t> next(32, 0);
    for (uint32_t i = 0; i < n && i < 64; ++i) {
        const uint8_t* e = elems + 32 * i;
        const char* name;
        uint32_t w[6];
        memcpy(&name, e, 8);
        memcpy(w, e + 8, 24);
        InputElement x{name ? name : "", w[0], w[1], w[2], w[3], w[4], w[5]};
        uint32_t slot = x.slot & 31;
        if (x.offset == 0xffffffffu) x.offset = next[slot];
        uint32_t bytes = 0, block = 1;
        texture::element(x.format, bytes, block);
        next[slot] = x.offset + bytes;
        (void)append;
        s.input.push_back(std::move(x));
    }
}

HRESULT finish(Pipeline* p, const Iid* iid, void** out) {
    stats::add(stats::Pipelines);
    exec::prepare(&p->state);
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

Pipeline* make() {
    auto* p = new Pipeline();
    p->object.functions = table();
    defaults(p->state);
    return p;
}

// The PC D3D12_GRAPHICS_PIPELINE_STATE_DESC.
HRESULT WINAPI create_graphics(void*, const GraphicsHead* d, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    auto* p = make();
    auto& s = p->state;
    s.root = d->root;
    for (uint32_t i = 0; i < 5; ++i) keep(p, i, d->stages[i]);
    auto* b = reinterpret_cast<const uint8_t*>(d);
    read_blend(s, b + 120);
    memcpy(&s.sample_mask, b + 448, 4);
    read_raster(s, b + 452);
    read_depth(s, b + 496, false);
    read_input(s, b + 552);
    memcpy(&s.strip_cut, b + 568, 4);
    memcpy(&s.topology_type, b + 572, 4);
    memcpy(&s.rt_count, b + 576, 4);
    memcpy(s.rt_formats, b + 580, 32);
    memcpy(&s.dsv_format, b + 612, 4);
    memcpy(&s.samples, b + 616, 4);
    return finish(p, iid, out);
}
HRESULT WINAPI create_compute(void*, const ComputeHead* d, const Iid* iid, void** out) {
    if (!d) return E_INVALIDARG;
    auto* p = make();
    p->state.compute = true;
    p->state.root = d->root;
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
    auto* p = make();
    auto& st = p->state;
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
        const uint8_t* q = d->stream + at;
        switch (kind) {
            case 0: memcpy(&st.root, q + 8, 8); break;
            case 8: read_blend(st, q + 4); break;
            case 9: memcpy(&st.sample_mask, q + 4, 4); break;
            case 10: read_raster(st, q + 4); break;
            case 11: read_depth(st, q + 4, false); break;
            case 21: read_depth(st, q + 4, true); break;
            case 12: read_input(st, q + 8); break;
            case 13: memcpy(&st.strip_cut, q + 4, 4); break;
            case 14: memcpy(&st.topology_type, q + 4, 4); break;
            case 15: memcpy(st.rt_formats, q + 4, 32), memcpy(&st.rt_count, q + 36, 4); break;
            case 16: memcpy(&st.dsv_format, q + 4, 4); break;
            case 17: memcpy(&st.samples, q + 4, 4); break;
            default: break;
        }
        if (stage >= 0) {
            Bytecode b;
            memcpy(&b, q + 8, sizeof b);
            keep(p, uint32_t(stage), b);
            if (stage == 5) st.compute = true;
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

State* state(void* obj) {
    auto* p = static_cast<Pipeline*>(obj);
    if (!p || p->object.functions != table()) return nullptr;
    return &p->state;
}

}  // namespace kl::gpu::pipeline
