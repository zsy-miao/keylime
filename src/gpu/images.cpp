// keylime GPU: title textures as Vulkan images (see images.h).
#include "gpu/images.h"

#include <windows.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <vector>

#include "gpu/host.h"
#include "gpu/memory.h"
#include "gpu/resource.h"
#include "gpu/retire.h"
#include "gpu/texture.h"
#include "log.h"
#include "stats.h"

namespace kl::gpu::images {
namespace {

constexpr uint32_t IMAGE_SLOTS = 65536, SAMPLER_SLOTS = 4096, STORAGE_SLOTS = 16384;
constexpr uint32_t TABLE_ENTRIES = 1u << 17;  // power of two; entries of 16 bytes after a 16-byte header
constexpr uint64_t CHUNK = 256ull << 20;
constexpr uint64_t PAGE4K = 4096;

// ------------------------------------------------------------ formats ----
struct Format {
    VkFormat vk;
    VkComponentSwizzle a = VK_COMPONENT_SWIZZLE_IDENTITY;  // alpha source (X8 formats: one)
    bool r_to_alpha = false;                               // A8: alpha from red, rgb zero
};

Format format_of(uint32_t f) {
    switch (f) {
        case 1: case 2: return {VK_FORMAT_R32G32B32A32_SFLOAT};
        case 3: return {VK_FORMAT_R32G32B32A32_UINT};
        case 4: return {VK_FORMAT_R32G32B32A32_SINT};
        case 5: case 6: return {VK_FORMAT_R32G32B32_SFLOAT};
        case 7: return {VK_FORMAT_R32G32B32_UINT};
        case 8: return {VK_FORMAT_R32G32B32_SINT};
        case 9: case 10: return {VK_FORMAT_R16G16B16A16_SFLOAT};
        case 11: return {VK_FORMAT_R16G16B16A16_UNORM};
        case 12: return {VK_FORMAT_R16G16B16A16_UINT};
        case 13: return {VK_FORMAT_R16G16B16A16_SNORM};
        case 14: return {VK_FORMAT_R16G16B16A16_SINT};
        case 15: case 16: return {VK_FORMAT_R32G32_SFLOAT};
        case 17: return {VK_FORMAT_R32G32_UINT};
        case 18: return {VK_FORMAT_R32G32_SINT};
        case 19: case 20: case 21: case 22: return {VK_FORMAT_D32_SFLOAT_S8_UINT};
        case 23: case 24: return {VK_FORMAT_A2B10G10R10_UNORM_PACK32};
        case 25: return {VK_FORMAT_A2B10G10R10_UINT_PACK32};
        case 26: return {VK_FORMAT_B10G11R11_UFLOAT_PACK32};
        case 27: case 28: return {VK_FORMAT_R8G8B8A8_UNORM};
        case 29: return {VK_FORMAT_R8G8B8A8_SRGB};
        case 30: return {VK_FORMAT_R8G8B8A8_UINT};
        case 31: return {VK_FORMAT_R8G8B8A8_SNORM};
        case 32: return {VK_FORMAT_R8G8B8A8_SINT};
        case 33: case 34: return {VK_FORMAT_R16G16_SFLOAT};
        case 35: return {VK_FORMAT_R16G16_UNORM};
        case 36: return {VK_FORMAT_R16G16_UINT};
        case 37: return {VK_FORMAT_R16G16_SNORM};
        case 38: return {VK_FORMAT_R16G16_SINT};
        case 39: case 41: return {VK_FORMAT_R32_SFLOAT};
        case 40: return {VK_FORMAT_D32_SFLOAT};
        case 42: return {VK_FORMAT_R32_UINT};
        case 43: return {VK_FORMAT_R32_SINT};
        case 44: case 45: case 46: case 47: return {VK_FORMAT_D24_UNORM_S8_UINT};
        case 48: case 49: return {VK_FORMAT_R8G8_UNORM};
        case 50: return {VK_FORMAT_R8G8_UINT};
        case 51: return {VK_FORMAT_R8G8_SNORM};
        case 52: return {VK_FORMAT_R8G8_SINT};
        case 53: case 54: return {VK_FORMAT_R16_SFLOAT};
        case 55: return {VK_FORMAT_D16_UNORM};
        case 56: return {VK_FORMAT_R16_UNORM};
        case 57: return {VK_FORMAT_R16_UINT};
        case 58: return {VK_FORMAT_R16_SNORM};
        case 59: return {VK_FORMAT_R16_SINT};
        case 60: case 61: return {VK_FORMAT_R8_UNORM};
        case 62: return {VK_FORMAT_R8_UINT};
        case 63: return {VK_FORMAT_R8_SNORM};
        case 64: return {VK_FORMAT_R8_SINT};
        case 65: return {VK_FORMAT_R8_UNORM, VK_COMPONENT_SWIZZLE_IDENTITY, true};
        case 67: return {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32};
        case 70: case 71: return {VK_FORMAT_BC1_RGBA_UNORM_BLOCK};
        case 72: return {VK_FORMAT_BC1_RGBA_SRGB_BLOCK};
        case 73: case 74: return {VK_FORMAT_BC2_UNORM_BLOCK};
        case 75: return {VK_FORMAT_BC2_SRGB_BLOCK};
        case 76: case 77: return {VK_FORMAT_BC3_UNORM_BLOCK};
        case 78: return {VK_FORMAT_BC3_SRGB_BLOCK};
        case 79: case 80: return {VK_FORMAT_BC4_UNORM_BLOCK};
        case 81: return {VK_FORMAT_BC4_SNORM_BLOCK};
        case 82: case 83: return {VK_FORMAT_BC5_UNORM_BLOCK};
        case 84: return {VK_FORMAT_BC5_SNORM_BLOCK};
        case 85: return {VK_FORMAT_R5G6B5_UNORM_PACK16};
        case 86: return {VK_FORMAT_A1R5G5B5_UNORM_PACK16};
        case 87: case 90: return {VK_FORMAT_B8G8R8A8_UNORM};
        case 88: case 92: return {VK_FORMAT_B8G8R8A8_UNORM, VK_COMPONENT_SWIZZLE_ONE};
        case 91: return {VK_FORMAT_B8G8R8A8_SRGB};
        case 93: return {VK_FORMAT_B8G8R8A8_SRGB, VK_COMPONENT_SWIZZLE_ONE};
        case 94: case 95: return {VK_FORMAT_BC6H_UFLOAT_BLOCK};
        case 96: return {VK_FORMAT_BC6H_SFLOAT_BLOCK};
        case 97: case 98: return {VK_FORMAT_BC7_UNORM_BLOCK};
        case 99: return {VK_FORMAT_BC7_SRGB_BLOCK};
        case 115: return {VK_FORMAT_B4G4R4A4_UNORM_PACK16};
        default: return {VK_FORMAT_UNDEFINED};
    }
}

bool depth_format(VkFormat f) {
    return f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D16_UNORM;
}
bool stencil_format(VkFormat f) { return f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D24_UNORM_S8_UINT; }

// A colour view format for sampling a depth resource through a colour SRV format.
VkImageAspectFlags view_aspect(VkFormat image, uint32_t dxgi_view) {
    if (!depth_format(image)) return VK_IMAGE_ASPECT_COLOR_BIT;
    // X24_TYPELESS_G8_UINT / X32_TYPELESS_G8X24_UINT read stencil.
    if (dxgi_view == 47 || dxgi_view == 22) return VK_IMAGE_ASPECT_STENCIL_BIT;
    return VK_IMAGE_ASPECT_DEPTH_BIT;
}

uint32_t format_size(VkFormat f) {  // bytes per texel (or block); 0 if unknown here
    switch (f) {
        case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
        case VK_FORMAT_BC2_UNORM_BLOCK: case VK_FORMAT_BC2_SRGB_BLOCK: case VK_FORMAT_BC3_UNORM_BLOCK: case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK: case VK_FORMAT_BC5_SNORM_BLOCK: case VK_FORMAT_BC6H_UFLOAT_BLOCK: case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK: case VK_FORMAT_BC7_SRGB_BLOCK: return 16;
        case VK_FORMAT_R32G32B32_SFLOAT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT: return 12;
        case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_UINT:
        case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT:
        case VK_FORMAT_R32G32_SINT: case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: case VK_FORMAT_BC4_UNORM_BLOCK:
        case VK_FORMAT_BC4_SNORM_BLOCK: return 8;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R16_SFLOAT:
        case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_SINT: case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_R5G6B5_UNORM_PACK16: case VK_FORMAT_A1R5G5B5_UNORM_PACK16: case VK_FORMAT_B4G4R4A4_UNORM_PACK16: return 2;
        case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_SINT: return 1;
        default: return 4;
    }
}

VkComponentMapping mapping_of(const Format& f) {
    VkComponentMapping m{};
    if (f.r_to_alpha) return {VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_ZERO, VK_COMPONENT_SWIZZLE_R};
    m.a = f.a;
    return m;
}

// ------------------------------------------------------------- memory ----
struct Chunk {
    VkDeviceMemory memory;
    uint64_t size, used;
    uint32_t type;
};
std::vector<Chunk> g_chunks;

bool bind_memory(VkImage image) {
    auto& h = host::get();
    VkMemoryRequirements req;
    vk::vkGetImageMemoryRequirements(h.device, image, &req);
    uint32_t type = h.caps.vram_type;
    if (type == ~0u || !(req.memoryTypeBits & (1u << type)))
        for (uint32_t i = 0; i < 32; ++i)
            if (req.memoryTypeBits & (1u << i)) {
                type = i;
                break;
            }
    for (auto& c : g_chunks) {
        if (c.type != type) continue;
        uint64_t at = (c.used + req.alignment - 1) & ~(req.alignment - 1);
        if (at + req.size <= c.size) {
            c.used = at + req.size;
            return vk::vkBindImageMemory(h.device, image, c.memory, at) == VK_SUCCESS;
        }
    }
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = std::max(CHUNK, req.size);
    ai.memoryTypeIndex = type;
    VkDeviceMemory m;
    if (!KL_VK_CHECK(vk::vkAllocateMemory(h.device, &ai, nullptr, &m))) return false;
    g_chunks.push_back({m, ai.allocationSize, req.size, type});
    return vk::vkBindImageMemory(h.device, image, m, 0) == VK_SUCCESS;
}

// ------------------------------------------------------------- images ----
}  // namespace

struct Image {
    uint64_t address = 0, size = 0;
    resource::ResourceDesc desc{};
    texture::Layout layout{};
    bool has_layout = false;
    uint32_t dxgi = 0, view_dxgi = 0;
    Format fmt{};
    VkImage image = VK_NULL_HANDLE;
    uint32_t width = 1, height = 1, layers = 1, mips = 1;
    bool depth = false, stencil = false, rt = false, storage = false;
    VkImageView sampled = VK_NULL_HANDLE, storage_view = VK_NULL_HANDLE;
    uint32_t slot = 0;
    bool initialized = false, dirty = true, gpu_owned = false, watched = false;
    std::map<uint64_t, VkImageView> targets;
};

namespace {

std::mutex g_lock;
std::map<uint64_t, Image*> g_images;     // by title address
std::vector<Image*> g_pending;           // to initialize / upload
uint32_t g_next_slot = 1, g_next_sampler = 1;
std::map<std::pair<uint32_t, uint32_t>, uint32_t> g_sampler_slots;
VkDescriptorSetLayout g_layout = VK_NULL_HANDLE;
VkDescriptorPool g_pool = VK_NULL_HANDLE;
VkDescriptorSet g_set = VK_NULL_HANDLE;
memory::Block g_table;
Image* g_dummy = nullptr;
bool g_started = false, g_watch = true;
std::vector<uint32_t> g_families;

// ----------------------------------------------------------- the table ----
uint32_t hash(uint32_t lo, uint32_t hi) { return (lo ^ (hi * 0x9e3779b1u)) & (TABLE_ENTRIES - 1); }

bool table_put(uint32_t lo, uint32_t hi, uint32_t slot) {
    auto* e = reinterpret_cast<volatile uint32_t*>(g_table.map + 16);
    uint32_t h = hash(lo, hi);
    for (uint32_t p = 0; p < 8; ++p) {
        uint32_t i = (h + p) & (TABLE_ENTRIES - 1);
        volatile uint32_t* w = e + 4 * i;
        bool empty = w[0] == 0 && w[1] == 0 && w[2] == 0;
        if (empty || (w[0] == lo && w[1] == hi)) {
            w[2] = slot;  // slot first: a reader matching the key sees it
            w[1] = hi;
            w[0] = lo;
            return true;
        }
    }
    static int warned = 0;
    if (warned++ < 8) KL_WARN("gpu", "texture table: no room near hash %x (key %x:%x)", h, lo, hi);
    return false;
}

// --------------------------------------------------------- descriptors ----
void write_image_slot(Image* im) {
    auto& h = host::get();
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, im->sampled, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2]{};
    w[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w[0].dstSet = g_set;
    w[0].dstBinding = 0;
    w[0].dstArrayElement = im->slot;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[0].pImageInfo = &ii;
    uint32_t n = 1;
    VkDescriptorImageInfo si{VK_NULL_HANDLE, im->storage_view, VK_IMAGE_LAYOUT_GENERAL};
    if (im->storage_view && im->slot < STORAGE_SLOTS) {
        w[1] = w[0];
        w[1].dstBinding = 2;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[1].pImageInfo = &si;
        n = 2;
    }
    vk::vkUpdateDescriptorSets(h.device, n, w, 0, nullptr);
}

VkImageView make_view(Image* im, VkFormat f, VkImageAspectFlags aspect, uint32_t mip, uint32_t mips, uint32_t layer,
                      VkComponentMapping map) {
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im->image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = f;
    vi.components = map;
    vi.subresourceRange = {aspect, mip, mips, layer, 1};
    VkImageView v = VK_NULL_HANDLE;
    KL_VK_CHECK(vk::vkCreateImageView(host::get().device, &vi, nullptr, &v));
    return v;
}

bool supports(VkFormat f, VkFormatFeatureFlags feat) {
    VkFormatProperties p{};
    vk::vkGetPhysicalDeviceFormatProperties(host::get().physical, f, &p);
    return (p.optimalTilingFeatures & feat) == feat;
}

// A storage-capable format of the same size class (sRGB -> UNORM).
VkFormat storage_format(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8G8B8A8_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
        case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
        default: return f;
    }
}

Image* create(uint64_t address, const resource::ResourceDesc& d) {
    auto& h = host::get();
    Format fmt = format_of(d.format);
    if (fmt.vk == VK_FORMAT_UNDEFINED || d.dimension == resource::BUFFER) return nullptr;
    auto* im = new Image();
    im->address = address;
    im->desc = d;
    im->dxgi = im->view_dxgi = d.format;
    im->fmt = fmt;
    im->has_layout = resource::texture_layout(d, im->layout);
    im->size = im->has_layout ? im->layout.size : 0;
    im->width = uint32_t(std::max<uint64_t>(1, d.width));
    im->height = d.dimension == resource::TEX1D ? 1 : std::max(1u, d.height);
    im->layers = std::max<uint32_t>(1, d.depth_or_array);  // 3D textures become arrays of their slices
    im->mips = im->has_layout ? uint32_t(im->layout.mips.size()) : std::max<uint32_t>(1, d.mip_levels);
    im->depth = depth_format(fmt.vk);
    im->stencil = stencil_format(fmt.vk);
    im->rt = (d.flags & resource::FLAG_RENDER_TARGET) || im->depth;
    // Block-compressed images: mips smaller than a block still need block-aligned copies (Vulkan handles it).
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt.vk;
    ci.extent = {im->width, im->height, 1};
    ci.mipLevels = im->mips;
    ci.arrayLayers = im->layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;  // MSAA resources are single-sampled for now
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (im->depth) {
        ci.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        ci.flags &= ~(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
    } else {
        if (supports(fmt.vk, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if ((d.flags & resource::FLAG_UNORDERED) && supports(storage_format(fmt.vk), VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
            ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
            im->storage = true;
        }
    }
    if (g_families.size() > 1) {
        ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = uint32_t(g_families.size());
        ci.pQueueFamilyIndices = g_families.data();
    }
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vk::vkCreateImage(h.device, &ci, nullptr, &im->image) != VK_SUCCESS || !bind_memory(im->image)) {
        static int warned = 0;
        if (warned++ < 8) KL_WARN("gpu", "could not make an image for %llux%u format %u", (unsigned long long)d.width, d.height, d.format);
        delete im;
        return nullptr;
    }
    VkImageAspectFlags aspect = im->depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    im->sampled = make_view(im, fmt.vk, aspect, 0, im->mips, 0, mapping_of(fmt));
    if (im->storage) im->storage_view = make_view(im, storage_format(fmt.vk), aspect, 0, 1, 0, {});
    im->gpu_owned = im->rt || (d.flags & resource::FLAG_UNORDERED);  // the GPU makes its contents
    im->dirty = !im->gpu_owned && im->has_layout;
    KL_TRACE("gpu", "image at %llx: %ux%ux%u, %u mips, format %u, flags %x, layout %x -> %s%s", (unsigned long long)address, im->width,
             im->height, im->layers, im->mips, d.format, d.flags, d.layout, im->gpu_owned ? "GPU made" : "uploaded",
             im->has_layout ? "" : " (no layout!)");
    stats::add(stats::Images);
    return im;
}

// The image at `address` for `d` (made, slotted and queued for its first upload when new).
Image* obtain(uint64_t address, const resource::ResourceDesc& d) {
    auto it = g_images.find(address);
    if (it != g_images.end()) {
        const auto& o = it->second->desc;
        if (o.width == d.width && o.height == d.height && o.depth_or_array == d.depth_or_array && o.dimension == d.dimension &&
            (o.format == d.format || format_of(o.format).vk == format_of(d.format).vk) &&
            (o.mip_levels == d.mip_levels || !d.mip_levels || !o.mip_levels))
            return it->second;
    }
    Image* im = create(address, d);
    if (!im) return nullptr;
    if (it != g_images.end()) {
        Image* old = it->second;
        it->second = im;
        im->slot = old->slot;  // the address keeps its slot; the old image goes when the GPU is done
        retire([old] {
            auto dev = host::get().device;
            for (auto& [k, v] : old->targets) vk::vkDestroyImageView(dev, v, nullptr);
            if (old->sampled) vk::vkDestroyImageView(dev, old->sampled, nullptr);
            if (old->storage_view) vk::vkDestroyImageView(dev, old->storage_view, nullptr);
            vk::vkDestroyImage(dev, old->image, nullptr);  // its memory stays in the chunk (bump allocator)
        });
    } else {
        g_images.emplace(address, im);
        if (g_next_slot >= IMAGE_SLOTS) {
            static bool warned = false;
            if (!warned) warned = true, KL_WARN("gpu", "out of image slots");
            im->slot = 0;
        } else {
            im->slot = g_next_slot++;
        }
    }
    if (im->slot) {
        write_image_slot(im);
        uint64_t a = address >> 8;
        table_put(uint32_t(a), uint32_t(a >> 32) & 0xff, im->slot);
    }
    g_pending.push_back(im);
    return im;
}

// -------------------------------------------------------- write watch ----
void protect(uint64_t start, uint64_t end, DWORD prot) {
    for (uint64_t at = start; at < end;) {
        MEMORY_BASIC_INFORMATION m;
        if (!VirtualQuery(reinterpret_cast<void*>(at), &m, sizeof m)) break;
        uint64_t rend = std::min<uint64_t>(end, uint64_t(m.BaseAddress) + m.RegionSize);
        if (m.State == MEM_COMMIT && (m.Protect == PAGE_READWRITE || m.Protect == PAGE_READONLY || m.Protect == PAGE_WRITECOMBINE ||
                                      m.Protect == (PAGE_READWRITE | PAGE_WRITECOMBINE) || m.Protect == (PAGE_READONLY | PAGE_WRITECOMBINE))) {
            DWORD old;
            DWORD want = prot | (m.Protect & PAGE_WRITECOMBINE);
            VirtualProtect(reinterpret_cast<void*>(at), size_t(rend - at), want, &old);
        }
        at = rend;
    }
}
uint64_t lo4k(uint64_t a) { return a & ~(PAGE4K - 1); }
uint64_t hi4k(uint64_t a) { return (a + PAGE4K - 1) & ~(PAGE4K - 1); }

// Opens [start, end) for writing and marks every image there changed. Lock held.
void open_range(uint64_t start, uint64_t end) {
    uint64_t s = start, e = end;
    // Every watched image touching the range opens entirely (so none keeps a half-open page).
    for (bool grew = true; grew;) {
        grew = false;
        auto it = g_images.lower_bound(s > (64ull << 20) ? s - (64ull << 20) : 0);
        for (; it != g_images.end() && it->first < e; ++it) {
            Image* im = it->second;
            uint64_t a = lo4k(im->address), b = hi4k(im->address + im->size);
            if (b <= s || a >= e) continue;
            if (im->watched) {
                im->watched = false;
                if (a < s) s = a, grew = true;
                if (b > e) e = b, grew = true;
            }
            if (!im->gpu_owned && !im->dirty && im->has_layout) {
                im->dirty = true;
                g_pending.push_back(im);
            }
        }
    }
    protect(lo4k(s), hi4k(e), PAGE_READWRITE);
}

LONG CALLBACK on_fault(EXCEPTION_POINTERS* x) {
    auto* r = x->ExceptionRecord;
    if (r->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || r->NumberParameters < 2 || r->ExceptionInformation[0] != 1)
        return EXCEPTION_CONTINUE_SEARCH;
    uint64_t a = r->ExceptionInformation[1];
    MEMORY_BASIC_INFORMATION m;
    if (!VirtualQuery(reinterpret_cast<void*>(a), &m, sizeof m) || (m.Protect & 0xff) != PAGE_READONLY) return EXCEPTION_CONTINUE_SEARCH;
    std::lock_guard g(g_lock);
    // Is it one of ours?
    bool ours = false;
    auto it = g_images.upper_bound(a);
    for (int n = 0; n < 64 && it != g_images.begin(); ++n) {
        --it;
        Image* im = it->second;
        if (im->address + im->size + PAGE4K > a && lo4k(im->address) <= a) {
            ours = true;
            break;
        }
        if (it->first + (1ull << 30) < a) break;
    }
    if (!ours) return EXCEPTION_CONTINUE_SEARCH;
    open_range(lo4k(a), lo4k(a) + PAGE4K);
    stats::add(stats::TextureWrites);
    return EXCEPTION_CONTINUE_EXECUTION;
}

// Memory the page table needs to import holds watched pages: open them first.
void before_import(uint64_t start, uint64_t end) {
    std::lock_guard g(g_lock);
    open_range(start, end);
}

// -------------------------------------------------------------- upload ----
struct Upload {
    Image* im;
    uint64_t at;
};

// Every page of [a, a + n) committed and readable (texture memory can be reserved before it is mapped).
bool readable(uint64_t a, uint64_t n) {
    for (uint64_t at = a; at < a + n;) {
        MEMORY_BASIC_INFORMATION m;
        if (!VirtualQuery(reinterpret_cast<void*>(at), &m, sizeof m) || m.State != MEM_COMMIT ||
            (m.Protect & (PAGE_NOACCESS | PAGE_GUARD)) || !m.Protect)
            return false;
        at = uint64_t(m.BaseAddress) + m.RegionSize;
    }
    return true;
}

// Detile, surviving memory that goes away under us.
bool detile_guarded(const texture::Layout& l, const void* surface, uint32_t mip, uint32_t slice, void* out, uint64_t row, uint64_t dp) {
    __try {
        return texture::detile(l, surface, mip, slice, out, row, dp);
    } __except (GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        return false;
    }
}

uint64_t linear_size(const Image* im) {
    const auto& l = im->layout;
    uint64_t total = 0;
    for (uint32_t m = 0; m < im->mips; ++m) {
        const auto& mip = l.mips[m];
        total += (uint64_t(mip.width_e) * mip.height_e * l.bytes_per_element + 15) & ~15ull;
    }
    return total * im->layers;
}

void barrier(VkCommandBuffer cb, const std::vector<VkImageMemoryBarrier2>& imgs, bool global) {
    VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    mb.srcAccessMask = mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    di.memoryBarrierCount = global ? 1 : 0;
    di.pMemoryBarriers = &mb;
    di.imageMemoryBarrierCount = uint32_t(imgs.size());
    di.pImageMemoryBarriers = imgs.data();
    if (di.memoryBarrierCount || di.imageMemoryBarrierCount) vk::vkCmdPipelineBarrier2(cb, &di);
}

VkImageAspectFlags all_aspects(const Image* im) {
    if (!im->depth) return VK_IMAGE_ASPECT_COLOR_BIT;
    return VK_IMAGE_ASPECT_DEPTH_BIT | (im->stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
}

}  // namespace

bool start() {
    std::lock_guard g(g_lock);
    if (g_started) return true;
    auto& h = host::get();
    {
        auto& q = h.queues;
        g_families = {q.graphics_family};
        for (uint32_t f : {q.compute_family, q.transfer_family})
            if (std::find(g_families.begin(), g_families.end(), f) == g_families.end()) g_families.push_back(f);
    }
    // Bindless set: all slots partially bound, updatable while in use.
    VkDescriptorSetLayoutBinding b[3] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, IMAGE_SLOTS, VK_SHADER_STAGE_ALL, nullptr},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, SAMPLER_SLOTS, VK_SHADER_STAGE_ALL, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, STORAGE_SLOTS, VK_SHADER_STAGE_ALL, nullptr},
    };
    VkDescriptorBindingFlags bf[3];
    for (auto& f : bf) f = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    bfi.bindingCount = 3;
    bfi.pBindingFlags = bf;
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.pNext = &bfi;
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    li.bindingCount = 3;
    li.pBindings = b;
    if (!KL_VK_CHECK(vk::vkCreateDescriptorSetLayout(h.device, &li, nullptr, &g_layout))) return false;
    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, IMAGE_SLOTS},
                                  {VK_DESCRIPTOR_TYPE_SAMPLER, SAMPLER_SLOTS},
                                  {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, STORAGE_SLOTS}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pi.maxSets = 1;
    pi.poolSizeCount = 3;
    pi.pPoolSizes = ps;
    if (!KL_VK_CHECK(vk::vkCreateDescriptorPool(h.device, &pi, nullptr, &g_pool))) return false;
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = g_pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &g_layout;
    if (!KL_VK_CHECK(vk::vkAllocateDescriptorSets(h.device, &ai, &g_set))) return false;
    // Texture table: header {mask}, then entries.
    if (!memory::make_block(16 + uint64_t(TABLE_ENTRIES) * 16, true, g_table)) return false;
    memset(g_table.map, 0, size_t(g_table.size));
    *reinterpret_cast<uint32_t*>(g_table.map) = TABLE_ENTRIES - 1;
    // Slot 0: what a lookup that misses gets (a transparent black texel, a linear wrapping sampler).
    resource::ResourceDesc dd{};
    dd.dimension = resource::TEX2D, dd.width = 1, dd.height = 1, dd.depth_or_array = 1, dd.mip_levels = 1, dd.format = 28;
    dd.sample_count = 1, dd.flags = resource::FLAG_RENDER_TARGET | resource::FLAG_UNORDERED;
    g_dummy = create(0, dd);
    if (g_dummy) {
        g_dummy->slot = 0;
        write_image_slot(g_dummy);
        g_pending.push_back(g_dummy);
    }
    uint32_t zero[4] = {0, 0, 0, 0};
    {
        // Sampler slot 0.
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.maxLod = VK_LOD_CLAMP_NONE;
        VkSampler s;
        if (KL_VK_CHECK(vk::vkCreateSampler(h.device, &si, nullptr, &s))) {
            VkDescriptorImageInfo ii{s, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = g_set, w.dstBinding = 1, w.descriptorCount = 1, w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, w.pImageInfo = &ii;
            vk::vkUpdateDescriptorSets(h.device, 1, &w, 0, nullptr);
        }
    }
    (void)zero;
    char v[4]{};
    if (GetEnvironmentVariableA("KEYLIME_TEXTURE_WATCH", v, 4) && v[0] == '0') g_watch = false;
    if (g_watch) AddVectoredExceptionHandler(1, on_fault);
    g_started = true;
    KL_INFO("gpu", "images ready: %u bindless image slots, %u samplers, texture table of %u entries%s", IMAGE_SLOTS,
            SAMPLER_SLOTS, TABLE_ENTRIES, g_watch ? ", texture writes watched" : "");
    return true;
}

Image* of_resource(void* res) {
    resource::Info r;
    if (!res || !resource::info(res, r) || r.desc.dimension == resource::BUFFER) return nullptr;
    std::lock_guard g(g_lock);
    if (!g_started) return nullptr;
    return obtain(r.address, r.desc);
}

void viewed(void* res, uint32_t format) {
    resource::Info r;
    if (!res || !resource::info(res, r) || r.desc.dimension == resource::BUFFER) return;
    std::lock_guard g(g_lock);
    if (!g_started) return;
    Image* im = obtain(r.address, r.desc);
    if (im && format && format != im->view_dxgi && !im->depth) {
        // A differently typed view (say sRGB of a typeless texture): sample through that format.
        Format f = format_of(format);
        if (f.vk != VK_FORMAT_UNDEFINED && format_size(f.vk) == format_size(im->fmt.vk) && f.vk != im->fmt.vk) {
            VkImageView v = make_view(im, f.vk, VK_IMAGE_ASPECT_COLOR_BIT, 0, im->mips, 0, mapping_of(f));
            if (v) {
                VkImageView old = im->sampled;
                im->sampled = v;
                im->view_dxgi = format;
                write_image_slot(im);
                retire([old] { vk::vkDestroyImageView(host::get().device, old, nullptr); });
            }
        }
    }
}

void viewed_placed(uint64_t address, const void* desc, uint32_t format) {
    if (!address || !desc) return;
    resource::ResourceDesc d;
    memcpy(&d, desc, sizeof d);
    if (d.dimension == resource::BUFFER) return;
    if (format) d.format = format;
    std::lock_guard g(g_lock);
    if (g_started) obtain(address, d);
}

void sampler(const uint32_t* w) {
    if (!w) return;
    uint32_t lo = w[0] ^ (w[2] * 0x85ebca6bu), hi = (w[1] ^ w[3]) | 0x80000000u;
    std::lock_guard g(g_lock);
    if (!g_started || g_sampler_slots.count({lo, hi})) return;
    if (g_next_sampler >= SAMPLER_SLOTS) return;
    auto addr = [](uint32_t m) {
        switch (m) {
            case 0: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case 1: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            case 2: case 4: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            case 3: case 5: case 7: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
            default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        }
    };
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.addressModeU = addr(w[0] & 7);
    si.addressModeV = addr((w[0] >> 3) & 7);
    si.addressModeW = addr((w[0] >> 6) & 7);
    uint32_t ratio = (w[0] >> 9) & 7;
    uint32_t mag = (w[2] >> 20) & 3, min = (w[2] >> 22) & 3, mip = (w[2] >> 26) & 3;
    si.magFilter = mag & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.minFilter = min & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = mip == 2 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    if ((mag >= 2 || min >= 2) && ratio) {
        si.anisotropyEnable = VK_TRUE;
        si.maxAnisotropy = float(1u << ratio);
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    }
    si.minLod = float(w[1] & 0xfff) / 256.0f;
    si.maxLod = mip == 0 ? 0.25f : float((w[1] >> 12) & 0xfff) / 256.0f;
    int bias = int(w[2] & 0x3fff);
    if (bias & 0x2000) bias -= 0x4000;
    si.mipLodBias = float(bias) / 256.0f;
    uint32_t border = w[3] >> 30;
    si.borderColor = border == 2 ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : border == 1 ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK
                                                                                     : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    VkSampler s;
    if (vk::vkCreateSampler(host::get().device, &si, nullptr, &s) != VK_SUCCESS) return;
    uint32_t slot = g_next_sampler++;
    g_sampler_slots[{lo, hi}] = slot;
    VkDescriptorImageInfo ii{s, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr.dstSet = g_set, wr.dstBinding = 1, wr.dstArrayElement = slot, wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, wr.pImageInfo = &ii;
    vk::vkUpdateDescriptorSets(host::get().device, 1, &wr, 0, nullptr);
    table_put(lo, hi, slot);
}

VkDescriptorSetLayout set_layout() { return g_layout; }
VkDescriptorSet set() { return g_set; }
uint64_t table() { return g_table.device; }

void prepare(VkCommandBuffer cb) {
    std::vector<Image*> work;
    {
        std::lock_guard g(g_lock);
        if (g_pending.empty()) return;
        work.swap(g_pending);
        static int n = 0;
        if (n++ < 3) KL_INFO("gpu", "prepare: %zu images pending", work.size());
    }
    std::vector<VkImageMemoryBarrier2> first;
    std::vector<Upload> ups;
    uint64_t total = 0;
    {
        std::lock_guard g(g_lock);
        std::sort(work.begin(), work.end());
        work.erase(std::unique(work.begin(), work.end()), work.end());
        for (Image* im : work) {
            if (!im->initialized) {
                im->initialized = true;
                VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
                b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
                b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = im->image;
                b.subresourceRange = {all_aspects(im), 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
                first.push_back(b);
            }
            if (im->dirty && !im->gpu_owned && im->has_layout && im->address) {
                if (!readable(im->address, im->size)) {
                    static int n = 0;
                    if (n++ < 6) {
                        MEMORY_BASIC_INFORMATION m{};
                        VirtualQuery(reinterpret_cast<void*>(im->address), &m, sizeof m);
                        KL_WARN("gpu", "texture at %llx (%llu bytes) not readable yet: state %lx protect %lx type %lx region %llx+%llx",
                                (unsigned long long)im->address, (unsigned long long)im->size, m.State, m.Protect, m.Type,
                                (unsigned long long)uint64_t(m.BaseAddress), (unsigned long long)m.RegionSize);
                    }
                    g_pending.push_back(im);  // not mapped yet: try again later
                    continue;
                }
                im->dirty = false;
                ups.push_back({im, total});
                total += linear_size(im);
            }
        }
    }
    barrier(cb, first, false);
    // Images the GPU makes start cleared (title memory starts zeroed too).
    for (auto& b : first) {
        Image* im = nullptr;
        for (Image* w : work)
            if (w->image == b.image) im = w;
        if (!im || !im->gpu_owned) continue;
        VkImageSubresourceRange r = b.subresourceRange;
        if (im->depth) {
            VkClearDepthStencilValue z{0.0f, 0};
            vk::vkCmdClearDepthStencilImage(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &r);
        } else {
            VkClearColorValue c{};
            vk::vkCmdClearColorImage(cb, im->image, VK_IMAGE_LAYOUT_GENERAL, &c, 1, &r);
        }
    }
    if (!ups.empty()) {
        memory::Block staging;
        if (!memory::make_block(total, false, staging)) {
            KL_WARN("gpu", "no staging memory for %llu MiB of texture uploads", (unsigned long long)(total >> 20));
            std::lock_guard g(g_lock);
            for (auto& u : ups) u.im->dirty = true, g_pending.push_back(u.im);
            return;
        }
        std::vector<VkBufferImageCopy> regions;
        for (auto& u : ups) {
            Image* im = u.im;
            if (g_watch) {
                std::lock_guard g(g_lock);
                // Watch from now on: a write after this point re-uploads (protect first, then read).
                protect(lo4k(im->address), hi4k(im->address + im->size), PAGE_READONLY);
                im->watched = true;
            }
            const auto& l = im->layout;
            uint64_t at = u.at;
            regions.clear();
            for (uint32_t s = 0; s < im->layers; ++s) {
                for (uint32_t m = 0; m < im->mips; ++m) {
                    const auto& mip = l.mips[m];
                    uint64_t row = uint64_t(mip.width_e) * l.bytes_per_element;
                    uint64_t bytes = (row * mip.height_e + 15) & ~15ull;
                    bool is3d = im->desc.dimension == resource::TEX3D;
                    // 3D textures: each slice is a layer; addrlib copies one slice at a time per call here.
                    bool ok = detile_guarded(l, reinterpret_cast<const void*>(im->address), m, is3d ? 0 : s, staging.map + at, row,
                                             row * mip.height_e);
                    if (ok && !is3d) {
                        VkBufferImageCopy c{};
                        c.bufferOffset = at;
                        c.bufferRowLength = mip.width_e * l.block;
                        c.bufferImageHeight = mip.height_e * l.block;
                        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, m, s, 1};
                        c.imageExtent = {std::max(1u, im->width >> m), std::max(1u, im->height >> m), 1};
                        regions.push_back(c);
                    }
                    at += bytes;
                }
            }
            if (!regions.empty())
                vk::vkCmdCopyBufferToImage(cb, staging.buffer, im->image, VK_IMAGE_LAYOUT_GENERAL, uint32_t(regions.size()), regions.data());
            stats::add(stats::TextureUploads);
        }
        memory::free_block(staging);  // retired after this submission
    }
    barrier(cb, {}, true);
}

Facts facts(Image* i) {
    return {i->image, i->fmt.vk, i->width, i->height, i->layers, i->mips, i->depth, i->stencil, i->rt};
}

VkImageView target_view(Image* im, uint32_t dxgi, uint32_t mip, uint32_t layer, VkFormat* out) {
    std::lock_guard g(g_lock);
    VkFormat f = im->fmt.vk;
    if (dxgi && !im->depth) {
        VkFormat v = format_of(dxgi).vk;
        if (v != VK_FORMAT_UNDEFINED && format_size(v) == format_size(f)) f = v;
    }
    mip = std::min(mip, im->mips - 1);
    layer = std::min(layer, im->layers - 1);
    if (out) *out = f;
    uint64_t key = uint64_t(f) << 32 | mip << 16 | layer;
    auto it = im->targets.find(key);
    if (it != im->targets.end()) return it->second;
    VkImageView v = make_view(im, f, all_aspects(im), mip, 1, layer, {});
    im->targets[key] = v;
    return v;
}

void gpu_wrote(Image* im) {
    if (!im || im->gpu_owned) return;
    std::lock_guard g(g_lock);
    im->gpu_owned = true;
    im->dirty = false;
}

void written(uint64_t address, uint64_t len) {
    std::lock_guard g(g_lock);
    if (!g_started || g_images.empty()) return;
    open_range(address, address + len);
}

VkFormat vk_format(uint32_t dxgi) { return format_of(dxgi).vk; }

const char* describe(uint64_t address) {
    std::lock_guard g(g_lock);
    auto it = g_images.find(address);
    if (it == g_images.end()) return "unknown";
    Image* im = it->second;
    return im->gpu_owned ? "GPU-made" : im->dirty ? "waiting for upload" : "uploaded";
}
bool is_depth_format(uint32_t dxgi) { return depth_format(format_of(dxgi).vk); }

// Called by memory before importing: see before_import.
void open_for_import(uint64_t start, uint64_t end) { before_import(start, end); }

}  // namespace kl::gpu::images
