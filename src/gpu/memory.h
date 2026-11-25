// keylime GPU: title memory as the GPU sees it.
//
// The console GPU reads and writes title memory at its own (CPU) addresses.
// keylime imports the title's committed memory into Vulkan as it is (host
// pointer import, zero copy), so GPU writes land where the title reads them
// and CPU writes need no upload. Translated shaders turn a title address into
// a device address through a two-level page table (1 GiB directory entries,
// 64 KiB pages) that this module keeps; pages that are not imported point at
// a scratch page, so a stray access reads zeros instead of faulting the GPU.
//
// Later, hot GPU-only pages can move to VRAM by changing their page table
// entries (the shaders never notice).
#pragma once
#include <cstdint>

#include "gpu/vk.h"

namespace kl::gpu::memory {

bool start();  // once, after host::start

// Makes the committed run of title memory holding [address, address + len) visible to the
// GPU; false if it is not committed memory or cannot be imported.
bool ensure(uint64_t address, uint64_t len = 1);
// The device buffer holding title address `address` (imported before), its offset there and
// how many bytes follow in it.
bool locate(uint64_t address, VkBuffer& buffer, uint64_t& offset, uint64_t& left);
// Title memory released (or about to be reused): its imports go.
void forget(uint64_t address, uint64_t len);

// Device address of the page table's directory (for shaders' push constants).
uint64_t page_table();

// Host-visible GPU memory for the host's own tables and staging (persistently mapped).
struct Block {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* map = nullptr;
    uint64_t device = 0, size = 0;
};
// prefer_vram: Resizable BAR memory when there is some (GPU reads fast, CPU writes combine).
bool make_block(uint64_t size, bool prefer_vram, Block& out, VkBufferUsageFlags extra = 0);
void free_block(Block& b);

struct Stats {
    uint64_t regions, bytes, failures;
};
Stats stats();

}  // namespace kl::gpu::memory
