// keylime GPU: PresentX -> a Vulkan swap chain on our window.
#pragma once
#include <mutex>

#include "gpu/vk.h"

namespace kl::gpu::present {
HRESULT present(VkQueue queue, std::mutex* queue_lock, uint32_t planes, const void* plane_list, const void* params);
}  // namespace kl::gpu::present
