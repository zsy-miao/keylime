// keylime GPU: d3d12_xs.dll entry points and the device object.
//
// The device is laid out as the title's inlined code expects: object header,
// a null-constant-buffer address, then "out of line" words. Byte 0x18 non-zero
// makes the title call our table instead of writing GPU packets itself, so
// every command reaches us as a call. Slot numbers of the device table follow
// the PC interface order (IUnknown 3, object 4, device 76 = 0x18..0x290, then
// Device1..Device8 additions 0x298..0x3a0 incl. the console's own, then the
// newer levels); the console-specific slot meanings were reverse engineered
// by VIctory (device.rs), whose map we reuse.
#include "gpu/device.h"

#include <mutex>

#include "gpu/exec.h"
#include "gpu/host.h"
#include "gpu/images.h"
#include "gpu/memory.h"
#include "hle/xmem.h"
#include "hle/hle.h"
#include "log.h"
#include "runtime.h"

namespace kl::gpu {
namespace {

constexpr HRESULT UNSUPPORTED = HRESULT(0x887A0004);
constexpr uint32_t CHECKED_VERSION = 0x00020004;  // D3D12.X interface 2.4 (the title passes it)
constexpr uint32_t HIGHEST_FEATURE_LEVEL = 0xc200;  // 12_2

constexpr Iid IID_DEVICE = guid(0x189819f1, 0x1db6, 0x4b57, {0xbe, 0x54, 0x18, 0x21, 0x33, 0x9b, 0x85, 0xf7});
// ID3D12Device1..8 and ID3D12Device11 (the title's).
const Iid LEVEL_IIDS[9] = {
    guid(0x77acce80, 0x638e, 0x4e65, {0x88, 0x95, 0xc1, 0xf2, 0x33, 0x86, 0x86, 0x3e}),
    guid(0x30baa41e, 0xb15b, 0x475c, {0xa0, 0xbb, 0x1a, 0xf5, 0xc5, 0xb6, 0x43, 0x28}),
    guid(0x81dadc15, 0x2bad, 0x4392, {0x93, 0xc5, 0x10, 0x13, 0x45, 0xc4, 0xaa, 0x98}),
    guid(0xe865df17, 0xa9ee, 0x46f9, {0xa4, 0x63, 0x30, 0x98, 0x31, 0x5a, 0xa2, 0xe5}),
    guid(0x8b4f173b, 0x2fea, 0x4b80, {0x8f, 0x58, 0x43, 0x07, 0x19, 0x1a, 0xb9, 0x5d}),
    guid(0xc70b221b, 0x40e4, 0x4a17, {0x89, 0xaf, 0x02, 0x5a, 0x07, 0x27, 0xa6, 0xdc}),
    guid(0x5c014b53, 0x68a1, 0x4b9b, {0x8b, 0xd1, 0xdd, 0x60, 0x46, 0xb9, 0x35, 0x8b}),
    guid(0x728f1fa2, 0xea11, 0x4a9f, {0x9b, 0x52, 0x9b, 0x42, 0xa4, 0x84, 0xd5, 0xc9}),
    guid(0x5405c344, 0xd457, 0x444e, {0xb4, 0xdd, 0x23, 0x66, 0xe4, 0x5a, 0xee, 0x39}),
};

struct Device {
    Object object;
    uint64_t null_constant_buffer;
    uint32_t out_of_line[16];
};
static_assert(sizeof(Device) == 0x58 && offsetof(Device, out_of_line) == 0x18);

void** table() {
    static void** t = make_table(Kind::Device, kMaxSlots);
    return t;
}

Device g_device{{nullptr, 0, {0}}, 0, {1}};
bool g_newer = false;

HRESULT WINAPI query_interface(Device*, const Iid* iid, void** out) {
    if (dxgi::is_device(iid)) return answer(Kind::Device, iid, dxgi::device(), out);
    return device_query(iid, out);
}
uint32_t WINAPI node_count(Device*) { return 1; }

// Feature query 0, as an RDNA 2 GPU reports it; the console's layout has one field more (index 11).
const uint32_t OPTIONS[16] = {1, 1, 2, 3, 3, 1, 1, 1, 3, 40, 0, 0, 0, 0, 1, 2};

struct FeatureLevels {
    uint32_t count;
    const uint32_t* requested;
    uint32_t highest;
};

HRESULT WINAPI check_feature_support(Device*, uint32_t feature, void* data, uint32_t size) {
    size_t expected;
    switch (feature) {
        case 0: expected = sizeof OPTIONS; break;
        case 1: expected = 16; break;
        case 2: expected = sizeof(FeatureLevels); break;
        case 3: expected = 12; break;
        case 4: expected = 16; break;
        case 7: expected = 4; break;
        case 8: expected = 24; break;
        case 23: expected = 12; break;
        default:
            KL_WARN("gpu", "device feature query %u (size %u) not implemented", feature, size);
            return E_NOTIMPL;
    }
    if (!data || size != expected) return E_INVALIDARG;
    auto* u = static_cast<uint32_t*>(data);
    switch (feature) {
        case 0: memcpy(data, OPTIONS, sizeof OPTIONS); return S_OK;
        case 1:  // architecture: one node, not tile-based, unified + cache-coherent
            if (u[0] != 0) return E_INVALIDARG;
            u[1] = 0, u[2] = 1, u[3] = 1;
            return S_OK;
        case 2: {
            auto* q = static_cast<FeatureLevels*>(data);
            if (!q->requested) return E_INVALIDARG;
            uint32_t best = 0;
            for (uint32_t i = 0; i < q->count; ++i)
                if (q->requested[i] <= HIGHEST_FEATURE_LEVEL && q->requested[i] > best) best = q->requested[i];
            if (!best) return UNSUPPORTED;
            q->highest = best;
            return S_OK;
        }
        case 3: u[1] = 0x7fffffff, u[2] = 0xffff; return S_OK;          // every use of every format
        case 4: u[3] = (u[1] == 1 || u[1] == 2 || u[1] == 4 || u[1] == 8); return S_OK;
        case 7: if (*u > 0x66) *u = 0x66; return S_OK;                    // shader model up to 6.6
        case 8: {                                                        // wave ops: 32..64 lanes
            uint32_t cus = runtime().console == Console::SeriesS ? 20 : 52;
            uint32_t v[6] = {1, 32, 64, cus * 64, 1, 1};
            memcpy(data, v, sizeof v);
            return S_OK;
        }
        case 23: u[0] = 0, u[1] = 0, u[2] = 1; return S_OK;               // native 16-bit ops
    }
    return E_NOTIMPL;
}

// D3D12XboxCreateDevice(adapter, parameters, iid, out). Parameters start with the interface version.
HRESULT WINAPI D3D12XboxCreateDevice(void*, const uint32_t* parameters, const Iid* iid, void** out) {
    if (!parameters || !iid) return E_INVALIDARG;
    KL_INFO("gpu", "D3D12.X device: interface %x.%x as %s", parameters[0] >> 16, parameters[0] & 0xffff,
            guid_text(iid).c_str());
    if (parameters[0] != CHECKED_VERSION) KL_WARN("gpu", "device layouts were checked against 2.4 only");
    if (!device_is(iid)) {
        KL_WARN("gpu", "device created as an unknown interface");
        return E_NOINTERFACE;
    }
    if (!out) return S_FALSE;
    g_newer = same(iid, LEVEL_IIDS[8]);
    static std::once_flag once;
    bool ok = true;
    std::call_once(once, [&] {
        ok = host::start() && memory::start() && images::start() && exec::start();
        if (ok) {
            xmem::set_write_hook([](uint64_t a, uint64_t n) { images::written(a, n); });
            // Other pages there now: imports of the old ones go, textures there reload.
            xmem::set_remap_hook([](uint64_t a, uint64_t n) {
                memory::remapped(a, n);
                images::written(a, n);
            });
        }
    });
    if (!ok || !host::ready()) {
        KL_ERROR("gpu", "no host GPU: the title cannot draw");
        return E_FAIL;
    }
    return answer(Kind::Device, iid, &g_device, out);
}

struct VirtualMemory {
    uint32_t flags, page_table_pages, preserved_bytes;
    uint64_t lowest, highest;
};
static_assert(sizeof(VirtualMemory) == 0x20);

// GPU addresses are CPU addresses (unified memory): the range is all of user space.
HRESULT WINAPI D3D12ConfigureVirtualMemoryX(VirtualMemory* c) {
    if (!c) return E_INVALIDARG;
    KL_INFO("gpu", "graphics virtual memory: flags %u, page table %u x 4 MiB, preserved %u bytes", c->flags,
            c->page_table_pages, c->preserved_bytes);
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    c->lowest = uint64_t(si.lpMinimumApplicationAddress);
    c->highest = uint64_t(si.lpMaximumApplicationAddress);
    return S_OK;
}

struct Init {
    Init() {
        void** t = table();
        fill_object(t, reinterpret_cast<void*>(&query_interface));
        t[7] = reinterpret_cast<void*>(&node_count);           // 0x38
        t[13] = reinterpret_cast<void*>(&check_feature_support); // 0x68
        t[69] = reinterpret_cast<void*>(&ignore);              // 0x228 debug error filter
        g_device.object.functions = t;
    }
} g_init;

}  // namespace

void device_set(size_t slot, void* fn) { table()[slot] = fn; }
void* device_object() { return &g_device; }
bool newer_layouts() { return g_newer; }

bool device_is(const Iid* iid) {
    if (same(iid, IID_DEVICE)) return true;
    for (auto& l : LEVEL_IIDS)
        if (same(iid, l)) return true;
    return false;
}

HRESULT device_query(const Iid* iid, void** out) {
    return answer(Kind::Device, iid, device_is(iid) ? &g_device : nullptr, out);
}

KL_MODULE("d3d12x", (std::vector<std::string>{"d3d12_xs.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(D3D12ConfigureVirtualMemoryX), KL_EXPORT(D3D12XboxCreateDevice)}));

}  // namespace kl::gpu
