// keylime GPU: PresentX -> a Vulkan swap chain on the title's window.
#pragma once
#include <windows.h>

#include <cstdint>

namespace kl::gpu::present {
HRESULT present(int queue, uint32_t planes, const void* plane_list, const void* params);
}  // namespace kl::gpu::present
