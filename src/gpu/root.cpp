// keylime GPU: root signatures. Serialization goes to the host d3d12.dll's
// serializer (the description is laid out as on PC, and its blob works as
// the console's). The console's own flags (bit 24 and up) are dropped first.
#include <windows.h>

#include <vector>

#include "gpu/device.h"
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

struct RootSignature {
    Object object;
    std::vector<uint8_t> blob;  // kept for translating pipelines
};
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
    auto* r = new RootSignature{{table(), 0, {0}}, std::vector<uint8_t>(blob, blob + size)};
    return query(r, iid, out);
}

struct Init {
    Init() { device_set(16, (void*)&create); }  // 0x80
} g_init;

}  // namespace

KL_MODULE("d3d12x-root", (std::vector<std::string>{"d3d12_xs.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(D3D12SerializeRootSignature), KL_EXPORT(D3D12SerializeVersionedRootSignature),
                                    KL_EXPORT(D3D12CreateRootSignatureDeserializer)}));

}  // namespace kl::gpu::root
