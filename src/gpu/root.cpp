// keylime GPU: root signatures. Serialization goes to the host d3d12.dll's
// serializer (the description is laid out as on PC, and its blob works as
// the console's). The console's own flags (bit 24 and up) are dropped first.
#include <windows.h>

#include <vector>

#include "gpu/descriptor.h"
#include "gpu/device.h"
#include "gpu/images.h"
#include "gpu/root.h"
#include "gpu/fence.h"
#include "hle/hle.h"
#include "log.h"

namespace kl::gpu::root {
namespace {

const Iid ROOT_IID = guid(0xc54a6b66, 0x72df, 0x4ee8, {0x8b, 0xe5, 0xa9, 0x46, 0xa1, 0x42, 0x92, 0x14});
constexpr uint32_t CONSOLE_FLAGS = 0xff000000;

struct Desc {
    uint32_t parameter_count;
    const void* parameters;
    uint32_t sampler_count;
    const void* samplers;
    uint32_t flags;
};
struct VersionedDesc {
    uint32_t version;
    Desc desc;
};

FARPROC host(const char* name) {
    static HMODULE m = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return m ? GetProcAddress(m, name) : nullptr;
}

HRESULT WINAPI D3D12SerializeRootSignature(const Desc* d, uint32_t version, void** blob, void** error) {
    if (!d) return E_INVALIDARG;
    using Fn = HRESULT(WINAPI*)(const Desc*, uint32_t, void**, void**);
    static auto f = reinterpret_cast<Fn>(host("D3D12SerializeRootSignature"));
    if (!f) return E_NOTIMPL;
    Desc c = *d;
    c.flags &= ~CONSOLE_FLAGS;
    return f(&c, version, blob, error);
}
HRESULT WINAPI D3D12SerializeVersionedRootSignature(const VersionedDesc* d, void** blob, void** error) {
    if (!d) return E_INVALIDARG;
    using Fn = HRESULT(WINAPI*)(const VersionedDesc*, void**, void**);
    static auto f = reinterpret_cast<Fn>(host("D3D12SerializeVersionedRootSignature"));
    if (!f) return E_NOTIMPL;
    VersionedDesc c = *d;
    c.desc.flags &= ~CONSOLE_FLAGS;
    HRESULT hr = f(&c, blob, error);
    if (FAILED(hr)) KL_WARN("gpu", "root signature serialization failed: %08lx", hr);
    return hr;
}
HRESULT WINAPI D3D12CreateRootSignatureDeserializer(const void* blob, size_t size, const Iid* iid, void** out) {
    using Fn = HRESULT(WINAPI*)(const void*, size_t, const Iid*, void**);
    static auto f = reinterpret_cast<Fn>(host("D3D12CreateRootSignatureDeserializer"));
    return f ? f(blob, size, iid, out) : E_NOTIMPL;
}

struct StaticSampler {
    uint32_t reg, space, words[4];
};
struct RootSignature {
    Object object;
    std::vector<uint8_t> blob;  // kept for translating pipelines
    std::vector<StaticSampler> samplers;
};

// The RTS0 part (the blob may be a DXBC container or the bare part).
bool rts0(const std::vector<uint8_t>& b, const uint8_t*& p, size_t& n) {
    if (b.size() >= 32 && !memcmp(b.data(), "DXBC", 4)) {
        uint32_t parts;
        memcpy(&parts, b.data() + 28, 4);
        for (uint32_t i = 0; i < parts && 32 + 4 * i + 4 <= b.size(); ++i) {
            uint32_t off, len;
            memcpy(&off, b.data() + 32 + 4 * i, 4);
            if (off + 8 > b.size()) continue;
            memcpy(&len, b.data() + off + 4, 4);
            if (!memcmp(b.data() + off, "RTS0", 4) && off + 8 + len <= b.size()) {
                p = b.data() + off + 8, n = len;
                return true;
            }
        }
        return false;
    }
    p = b.data(), n = b.size();
    return n >= 24;
}

void read_samplers(RootSignature* r) {
    const uint8_t* p;
    size_t n;
    if (!rts0(r->blob, p, n)) return;
    uint32_t h[6];
    memcpy(h, p, 24);
    uint32_t version = h[0], count = h[3], at = h[4];
    size_t each = version >= 3 ? 14 * 4 : 13 * 4;
    for (uint32_t i = 0; i < count && at + (i + 1) * each <= n; ++i) {
        uint32_t d[14];
        memcpy(d, p + at + i * each, each);
        StaticSampler s{d[10], d[11], {}};
        descriptor::sampler_from_static(d, s.words);
        images::sampler(s.words);
        r->samplers.push_back(s);
    }
}
HRESULT WINAPI query(RootSignature* r, const Iid* iid, void** out) {
    bool ok = same(iid, ROOT_IID) || same(iid, fence::PAGEABLE_IIDS[1]) || same(iid, fence::PAGEABLE_IIDS[2]);
    return answer(Kind::RootSignature, iid, ok ? r : nullptr, out);
}
void** table() {
    static void** t = [] {
        void** t = make_table(Kind::RootSignature, 24);
        fill_object(t, (void*)&query);
        fill_child(t);
        return t;
    }();
    return t;
}
HRESULT WINAPI create(void*, uint32_t, const uint8_t* blob, size_t size, const Iid* iid, void** out) {
    if (!blob || !size) return E_INVALIDARG;
    auto* r = new RootSignature{{table(), 0, {0}}, std::vector<uint8_t>(blob, blob + size), {}};
    read_samplers(r);
    return query(r, iid, out);
}

struct Init {
    Init() { device_set(16, (void*)&create); }  // 0x80
} g_init;

}  // namespace

bool static_sampler(void* obj, uint32_t reg, uint32_t space, uint32_t* words) {
    auto* r = static_cast<RootSignature*>(obj);
    if (!r || r->object.functions != table()) return false;
    for (auto& s : r->samplers)
        if (s.reg == reg && s.space == space) return memcpy(words, s.words, 16), true;
    for (auto& s : r->samplers)
        if (s.reg == reg) return memcpy(words, s.words, 16), true;
    return false;
}

KL_MODULE("d3d12x-root", (std::vector<std::string>{"d3d12_xs.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(D3D12SerializeRootSignature), KL_EXPORT(D3D12SerializeVersionedRootSignature),
                                    KL_EXPORT(D3D12CreateRootSignatureDeserializer)}));

}  // namespace kl::gpu::root
