// keylime GPU: command allocators and lists (recording into Vulkan command buffers).
#pragma once
#include "gpu/vk.h"

namespace kl::gpu::list {
// The host command buffer a title list recorded, closed for submission (null if none).
VkCommandBuffer close_for_submit(void* list);
// The list was submitted on `queue`: its allocator may recycle after the GPU is done.
void submitted(void* list, VkQueue queue);
}  // namespace kl::gpu::list
