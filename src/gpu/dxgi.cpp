// keylime GPU: the DXGI side of the device: one adapter with one output,
// reached from the device by interface query. Tables follow the PC
// interfaces' order. IIDs and slots as VIctory found them (dxgi.rs).
#include "gpu/device.h"
#include "gpu/host.h"
#include "hle/xmem.h"
#include "log.h"

namespace kl::gpu::dxgi {
namespace {

const Iid DEVICE_IIDS[3] = {guid(0x54ec77fa, 0x1377, 0x44e6, {0x8c, 0x32, 0x88, 0xfd, 0x5f, 0x44, 0xc8, 0x4c}),
                            guid(0x77db970f, 0x6276, 0x48ba, {0xba, 0x28, 0x07, 0x01, 0x43, 0xb4, 0x39, 0x2c}),
                            guid(0x05008617, 0xfbfd, 0x4051, {0xa7, 0x90, 0x14, 0x48, 0x84, 0xb4, 0xf6, 0xa9})};
const Iid ADAPTER_IIDS[3] = {guid(0x2411e7e1, 0x12ac, 0x4ccf, {0xbd, 0x14, 0x97, 0x98, 0xe8, 0x53, 0x4d, 0xc0}),
                             guid(0x29038f61, 0x3839, 0x4626, {0x91, 0xfd, 0x08, 0x68, 0x79, 0x01, 0x1a, 0x05}),
                             guid(0x0aa1ae0a, 0xfa0e, 0x4b84, {0x86, 0x44, 0xe0, 0x5f, 0xf8, 0xe5, 0xac, 0xb5})};
const Iid FACTORY_IIDS[3] = {guid(0x7b7166ec, 0x21c7, 0x44ae, {0xb2, 0x1a, 0xc9, 0xae, 0x32, 0x1a, 0xe3, 0x69}),
                             guid(0x770aae78, 0xf26f, 0x4dba, {0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87}),
                             guid(0x50c83a1c, 0xe072, 0x4c48, {0x87, 0xb0, 0x36, 0x30, 0xfa, 0x36, 0xa6, 0xd0})};
const Iid OUTPUT_IID = guid(0xae02eedb, 0xc735, 0x4690, {0x8d, 0x52, 0x5a, 0x8d, 0xc2, 0x02, 0x13, 0xaa});

template <size_t N>
bool in(const Iid* iid, const Iid (&set)[N]) {
    for (auto& g : set)
        if (same(iid, g)) return true;
    return false;
}

Object g_device{}, g_adapter{}, g_factory{}, g_output{};

HRESULT WINAPI device_query_m(Object* o, const Iid* iid, void** out) {
    if (device_is(iid)) return device_query(iid, out);
    return answer(Kind::Dxgi, iid, in(iid, DEVICE_IIDS) ? o : nullptr, out);
}
HRESULT WINAPI device_parent(Object*, const Iid* iid, void** out) {
    return answer(Kind::Dxgi, iid, in(iid, ADAPTER_IIDS) ? &g_adapter : nullptr, out);
}
HRESULT WINAPI device_adapter(Object*, void** out) { return answer(Kind::Dxgi, &ADAPTER_IIDS[0], &g_adapter, out); }
HRESULT WINAPI adapter_query(Object* o, const Iid* iid, void** out) { return answer(Kind::Dxgi, iid, in(iid, ADAPTER_IIDS) ? o : nullptr, out); }
HRESULT WINAPI adapter_parent(Object*, const Iid* iid, void** out) {
    return answer(Kind::Dxgi, iid, in(iid, FACTORY_IIDS) ? &g_factory : nullptr, out);
}
HRESULT WINAPI adapter_outputs(Object*, uint32_t index, void** out) {
    if (index) return DXGI_ERROR_NOT_FOUND;
    return answer(Kind::Dxgi, &OUTPUT_IID, &g_output, out);
}
struct AdapterDescription {
    wchar_t name[128];
    uint32_t vendor, device, subsystem, revision;
    size_t dedicated_video, dedicated_system, shared_system;
    uint32_t luid[2];
};
static_assert(sizeof(AdapterDescription) == 0x130);
HRESULT WINAPI adapter_description(Object*, AdapterDescription* out) {
    if (!out) return E_INVALIDARG;
    *out = {};
    wcscpy_s(out->name, L"keylime console GPU");
    out->vendor = 0x1002;  // the title expects the console's AMD part
    size_t gpu, standard;
    xmem::title_memory(gpu, standard);
    out->dedicated_video = gpu;
    out->shared_system = standard;
    out->luid[0] = 1;
    return S_OK;
}
HRESULT WINAPI factory_query(Object* o, const Iid* iid, void** out) { return answer(Kind::Dxgi, iid, in(iid, FACTORY_IIDS) ? o : nullptr, out); }
HRESULT WINAPI factory_adapters(Object*, uint32_t index, void** out) {
    if (index) return DXGI_ERROR_NOT_FOUND;
    return answer(Kind::Dxgi, &ADAPTER_IIDS[0], &g_adapter, out);
}
HRESULT WINAPI output_query(Object* o, const Iid* iid, void** out) { return answer(Kind::Dxgi, iid, same(iid, OUTPUT_IID) ? o : nullptr, out); }
HRESULT WINAPI output_parent(Object*, const Iid* iid, void** out) {
    return answer(Kind::Dxgi, iid, in(iid, ADAPTER_IIDS) ? &g_adapter : nullptr, out);
}
struct Mode {
    uint32_t width, height, refresh[2], format, scanlines, scaling, flags;
};
// The console's mode list: one 4K mode. The title looks for 120 Hz to allow 120 Hz frame
// intervals; we offer 60 Hz until presentation keeps up.
HRESULT WINAPI modes(Object*, uint32_t, uint32_t, uint32_t* count, Mode* out) {
    if (!count) return E_INVALIDARG;
    if (!out) return *count = 1, S_OK;
    if (*count < 1) return DXGI_ERROR_MORE_DATA;
    *count = 1;
    *out = {3840, 2160, {60, 1}, 0, 1, 0, 0};
    return S_OK;
}

void** table(void* query, std::initializer_list<std::pair<int, void*>> slots) {
    void** t = make_table(Kind::Dxgi, 40);
    fill_object(t, query);
    for (auto& [s, f] : slots) t[s] = f;
    return t;
}

struct Init {
    Init() {
        g_device.functions = table((void*)&device_query_m, {{6, (void*)&device_parent}, {7, (void*)&device_adapter}});
        g_adapter.functions = table((void*)&adapter_query, {{6, (void*)&adapter_parent}, {7, (void*)&adapter_outputs},
                                                            {8, (void*)&adapter_description}});
        g_factory.functions = table((void*)&factory_query, {{7, (void*)&factory_adapters}, {12, (void*)&factory_adapters}});
        g_output.functions = table((void*)&output_query, {{6, (void*)&output_parent}, {21, (void*)&modes}});
    }
} g_init;

}  // namespace

void* device() { return &g_device; }
bool is_device(const Iid* iid) { return in(iid, DEVICE_IIDS); }

}  // namespace kl::gpu::dxgi
