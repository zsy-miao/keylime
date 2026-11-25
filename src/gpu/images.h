// keylime GPU: title textures as Vulkan images.
//
// A texture's contents live in title memory in the console's tiled layout;
// its Vulkan image (device-local) is a copy the host keeps current: written
// by the GPU when it renders or copies into it, and re-uploaded (detiled with
// AMD's addrlib) when the title changes the memory. Changes are seen through
// write protection of uploaded texture memory (one fault per page, then the
// pages are open until the next upload) and through notices from the
// streaming code, which writes through its own mappings.
//
// Shaders find images and samplers through the texture table: a GPU hash of
// the T# base address (or the S# words) to a slot in the bindless arrays.
// Every image is in VK_IMAGE_LAYOUT_GENERAL for its whole life.
#pragma once
#include <cstdint>

#include "gpu/vk.h"

namespace kl::gpu::images {

struct Image;

bool start();  // once, after memory::start

// The image of a title texture resource (made on first use); null for buffers / unknown formats.
Image* of_resource(void* resource);
// A shader resource / unordered view of `resource` with `format` (0: the resource's) was made:
// shaders may now find the image by its address.
void viewed(void* resource, uint32_t format);
void viewed_placed(uint64_t address, const void* resource_desc, uint32_t format);
// A sampler descriptor's words (S#): shaders may now find a matching Vulkan sampler.
void sampler(const uint32_t* words);

VkDescriptorSetLayout set_layout();
VkDescriptorSet set();
uint64_t table();  // device address of the texture table

// Records uploads of changed images and first-use layout changes (outside rendering).
void prepare(VkCommandBuffer cb);

struct Facts {
    VkImage image;
    VkFormat format;
    uint32_t width, height, depth_or_layers, mips;
    bool depth, stencil, render_target;
};
Facts facts(Image* i);
// A view for rendering into (mip, layer) as `dxgi_format` (0: the image's).
VkImageView target_view(Image* i, uint32_t dxgi_format, uint32_t mip, uint32_t layer, VkFormat* format = nullptr);
// The page table is about to import [start, end): open watched texture pages there.
void open_for_import(uint64_t start, uint64_t end);
// The GPU wrote the image (rendering, copies): its memory copy is stale, keep the image.
void gpu_wrote(Image* i);

// Title memory changed without a CPU write we could see (streaming writes through other mappings).
void written(uint64_t address, uint64_t len);

VkFormat vk_format(uint32_t dxgi);
const char* describe(uint64_t address);  // diagnostics: what the image at a T# base address is
bool is_depth_format(uint32_t dxgi);

}  // namespace kl::gpu::images
