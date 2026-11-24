// keylime: d3d12_xs.dll entry points. The device itself is keylime's own GPU
// layer (to come); for now only what the title calls before creating it.
#include <windows.h>

#include "hle/hle.h"
#include "log.h"

namespace kl::gpu {
namespace {

struct VirtualMemory {
    uint32_t flags;
    uint32_t page_table_pages;  // GPU page table size in 4 MiB pages
    uint32_t preserved_bytes;
    uint64_t lowest, highest;   // out: the range graphics addresses fall in
};
static_assert(sizeof(VirtualMemory) == 0x20);

// GPU addresses are CPU addresses here: the range is all of user space.
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

}  // namespace

KL_MODULE("d3d12x", (std::vector<std::string>{"d3d12_xs.dll"}),
          (std::vector<hle::Export>{KL_EXPORT(D3D12ConfigureVirtualMemoryX)}));

}  // namespace kl::gpu
