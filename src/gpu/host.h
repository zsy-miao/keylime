// keylime GPU: the host side, on Vulkan.
//
// Design (ours, not VIctory's):
//  - Title GPU addresses are CPU addresses. The host sees title memory through
//    buffers with device addresses, so translated shaders keep RDNA's raw
//    64-bit pointers (buffer_device_address) instead of descriptor rewriting.
//  - Ownership decides placement: GPU-only data lives in VRAM; CPU-written data
//    goes through Resizable BAR (host-visible VRAM); CPU readback of GPU data
//    happens on demand (page fault), not by watching every page.
//  - Compute shaders may run as CUDA kernels (VK_NVX_binary_import) where
//    that beats the SPIR-V path; wave64 = one 32-lane warp, two lanes per thread.
#pragma once
#include <cstdint>
#include <string>

#include "gpu/vk.h"

namespace kl::gpu::host {

struct Queues {
    uint32_t graphics_family = ~0u, compute_family = ~0u, transfer_family = ~0u;
    VkQueue graphics = VK_NULL_HANDLE, compute = VK_NULL_HANDLE, transfer = VK_NULL_HANDLE;
};

struct Caps {
    bool buffer_address = false, mesh_shader = false, host_memory_import = false, cuda_binary = false;
    bool ray_tracing = false, descriptor_buffer = false, rebar = false, int64_atomics = false, memory_budget = false, cache_control = false, checkpoints = false, device_fault = false;
    uint64_t rebar_bytes = 0, vram_bytes = 0;
    uint32_t vram_type = ~0u, rebar_type = ~0u, host_type = ~0u, readback_type = ~0u;
    uint64_t host_import_alignment = 0;
};

struct Host {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memory{};
    Queues queues;
    Caps caps;
    std::string name;
};

bool start();   // once, at device creation
bool ready();
Host& get();

// What the driver lets this process keep in VRAM (the device-local heap), and how much it holds now.
void vram(uint64_t& budget, uint64_t& usage);

}  // namespace kl::gpu::host
