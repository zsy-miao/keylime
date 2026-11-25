// keylime GPU: title memory imported into Vulkan, and the GPU page table.
#include "gpu/memory.h"

#include <windows.h>

#include <map>
#include <mutex>
#include <vector>

#include "gpu/host.h"
#include "gpu/images.h"
#include "gpu/retire.h"
#include "log.h"

namespace kl::gpu::memory {
namespace {

constexpr uint64_t PAGE = 64 << 10, GIB = 1ull << 30;
constexpr uint64_t DIRECTORY = 1ull << 17;     // 2^47 bytes of address space / 1 GiB
constexpr uint64_t TABLE = GIB / PAGE;          // 16384 entries per 1 GiB table
constexpr uint64_t TABLE_BYTES = TABLE * 8;     // 128 KiB
constexpr uint64_t POOL_TABLES = 256;           // tables per pool block (32 MiB)

struct Region {
    uint64_t end;
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint64_t device;
};

std::mutex g_lock;
std::map<uint64_t, Region> g_regions;  // by start
Block g_dir, g_scratch;
std::vector<Block> g_pools;
uint32_t g_pool_used = POOL_TABLES;  // tables used in the last pool
uint64_t* g_tables[DIRECTORY] = {};   // host view of each directory entry's table (null: the empty one)
uint64_t g_empty_table_device = 0;
uint64_t* g_empty_table = nullptr;
Stats g_stats{};
bool g_started = false;

std::vector<uint32_t> families() {
    auto& q = host::get().queues;
    std::vector<uint32_t> f = {q.graphics_family};
    for (uint32_t x : {q.compute_family, q.transfer_family}) {
        bool seen = false;
        for (uint32_t y : f) seen |= x == y;
        if (!seen) f.push_back(x);
    }
    return f;
}

void share(VkBufferCreateInfo& bi, const std::vector<uint32_t>& f) {
    if (f.size() > 1) {
        bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bi.queueFamilyIndexCount = uint32_t(f.size());
        bi.pQueueFamilyIndices = f.data();
    }
}

constexpr VkBufferUsageFlags ALL_USES = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

uint64_t address_of(VkBuffer b) {
    VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    ai.buffer = b;
    return vk::vkGetBufferDeviceAddress(host::get().device, &ai);
}

// A table for directory entry `d` (made on first use).
uint64_t* table_for(uint64_t d) {
    if (g_tables[d]) return g_tables[d];
    if (g_pool_used == POOL_TABLES) {
        Block b;
        if (!make_block(POOL_TABLES * TABLE_BYTES, true, b)) return nullptr;
        g_pools.push_back(b);
        g_pool_used = 0;
    }
    Block& b = g_pools.back();
    uint64_t off = uint64_t(g_pool_used++) * TABLE_BYTES;
    auto* t = reinterpret_cast<uint64_t*>(b.map + off);
    for (uint64_t i = 0; i < TABLE; ++i) t[i] = g_scratch.device;
    reinterpret_cast<uint64_t*>(g_dir.map)[d] = b.device + off;
    return g_tables[d] = t;
}

// Points the pages of [start, end) at device address `device` (start's), or at scratch if 0.
void map_pages(uint64_t start, uint64_t end, uint64_t device) {
    for (uint64_t p = start & ~(PAGE - 1); p < end; p += PAGE) {
        uint64_t d = p >> 30;
        if (d >= DIRECTORY) break;
        uint64_t* t = device ? table_for(d) : g_tables[d];
        if (!t) continue;
        t[(p >> 16) & (TABLE - 1)] = device ? device + (p - start) : g_scratch.device;  // wraps for a partial first page
    }
}

void drop(std::map<uint64_t, Region>::iterator it) {
    Region r = it->second;
    uint64_t start = it->first;
    g_stats.regions--, g_stats.bytes -= r.end - start;
    g_regions.erase(it);
    retire([r] {
        auto dev = host::get().device;
        vk::vkDestroyBuffer(dev, r.buffer, nullptr);
        vk::vkFreeMemory(dev, r.memory, nullptr);
    });
}

bool import(uint64_t start, uint64_t end, Region& out) {
    auto& h = host::get();
    uint64_t len = end - start;
    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (vk::vkGetMemoryHostPointerPropertiesEXT(h.device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                                reinterpret_cast<void*>(start), &hp) != VK_SUCCESS || !hp.memoryTypeBits)
        return false;
    static const std::vector<uint32_t> fam = families();
    VkExternalMemoryBufferCreateInfo ebi{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ebi.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.pNext = &ebi;
    bi.size = len;
    bi.usage = ALL_USES;
    share(bi, fam);
    VkBuffer buf;
    if (vk::vkCreateBuffer(h.device, &bi, nullptr, &buf) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vk::vkGetBufferMemoryRequirements(h.device, buf, &req);
    uint32_t bits = req.memoryTypeBits & hp.memoryTypeBits, type = ~0u;
    if (h.caps.host_type != ~0u && (bits & (1u << h.caps.host_type))) type = h.caps.host_type;
    for (uint32_t i = 0; i < 32 && type == ~0u; ++i)
        if (bits & (1u << i)) type = i;
    if (type == ~0u) {
        vk::vkDestroyBuffer(h.device, buf, nullptr);
        return false;
    }
    VkMemoryAllocateFlagsInfo fl{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.pNext = &fl;
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = reinterpret_cast<void*>(start);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &imp;
    ai.allocationSize = len;
    ai.memoryTypeIndex = type;
    VkDeviceMemory mem;
    VkResult r = vk::vkAllocateMemory(h.device, &ai, nullptr, &mem);
    if (r != VK_SUCCESS) {
        vk::vkDestroyBuffer(h.device, buf, nullptr);
        return false;
    }
    vk::vkBindBufferMemory(h.device, buf, mem, 0);
    out = {end, buf, mem, address_of(buf)};
    return true;
}

// The committed run of the allocation holding `a`: [start, end).
bool committed_run(uint64_t a, uint64_t& start, uint64_t& end) {
    MEMORY_BASIC_INFORMATION m;
    auto usable = [](const MEMORY_BASIC_INFORMATION& m) {
        return m.State == MEM_COMMIT && !(m.Protect & (PAGE_GUARD | PAGE_NOACCESS)) && m.Protect;
    };
    if (!VirtualQuery(reinterpret_cast<void*>(a), &m, sizeof m) || !usable(m)) return false;
    void* base = m.AllocationBase;
    start = uint64_t(m.BaseAddress), end = start + m.RegionSize;
    while (start > uint64_t(base)) {
        MEMORY_BASIC_INFORMATION p;
        if (!VirtualQuery(reinterpret_cast<void*>(start - 1), &p, sizeof p) || p.AllocationBase != base || !usable(p)) break;
        start = uint64_t(p.BaseAddress);
    }
    for (;;) {
        MEMORY_BASIC_INFORMATION n;
        if (!VirtualQuery(reinterpret_cast<void*>(end), &n, sizeof n) || n.AllocationBase != base || !usable(n)) break;
        end = uint64_t(n.BaseAddress) + n.RegionSize;
    }
    return true;
}

std::map<uint64_t, Region>::iterator holding(uint64_t a) {
    auto it = g_regions.upper_bound(a);
    if (it == g_regions.begin()) return g_regions.end();
    --it;
    return a < it->second.end ? it : g_regions.end();
}

}  // namespace

bool make_block(uint64_t size, bool prefer_vram, Block& out, VkBufferUsageFlags extra) {
    auto& h = host::get();
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = ALL_USES | extra;
    static const std::vector<uint32_t> fam = families();
    share(bi, fam);
    if (vk::vkCreateBuffer(h.device, &bi, nullptr, &out.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    vk::vkGetBufferMemoryRequirements(h.device, out.buffer, &req);
    uint32_t type = prefer_vram && h.caps.rebar && (req.memoryTypeBits & (1u << h.caps.rebar_type)) ? h.caps.rebar_type : h.caps.host_type;
    VkMemoryAllocateFlagsInfo fl{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    fl.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.pNext = &fl;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type;
    if (!KL_VK_CHECK(vk::vkAllocateMemory(h.device, &ai, nullptr, &out.memory))) {
        vk::vkDestroyBuffer(h.device, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return false;
    }
    vk::vkBindBufferMemory(h.device, out.buffer, out.memory, 0);
    void* p = nullptr;
    vk::vkMapMemory(h.device, out.memory, 0, VK_WHOLE_SIZE, 0, &p);
    out.map = static_cast<uint8_t*>(p);
    out.size = size;
    out.device = address_of(out.buffer);
    return out.map != nullptr;
}

void free_block(Block& b) {
    if (!b.buffer) return;
    Block c = b;
    retire([c] {
        auto dev = host::get().device;
        vk::vkDestroyBuffer(dev, c.buffer, nullptr);
        vk::vkFreeMemory(dev, c.memory, nullptr);
    });
    b = {};
}

bool start() {
    std::lock_guard g(g_lock);
    if (g_started) return true;
    auto& h = host::get();
    if (!h.caps.host_memory_import) {
        KL_ERROR("gpu", "the driver cannot import host memory (VK_EXT_external_memory_host): the GPU cannot see title memory");
        return false;
    }
    // Scratch: where unmapped pages lead (one page plus room for a wide access past its end).
    if (!make_block(2 * PAGE, true, g_scratch) || !make_block(DIRECTORY * 8, true, g_dir)) return false;
    memset(g_scratch.map, 0, size_t(g_scratch.size));
    // The empty table every directory entry starts with.
    Block empty;
    if (!make_block(TABLE_BYTES, true, empty)) return false;
    g_empty_table = reinterpret_cast<uint64_t*>(empty.map);
    g_empty_table_device = empty.device;
    for (uint64_t i = 0; i < TABLE; ++i) g_empty_table[i] = g_scratch.device;
    auto* dir = reinterpret_cast<uint64_t*>(g_dir.map);
    for (uint64_t i = 0; i < DIRECTORY; ++i) dir[i] = g_empty_table_device;
    g_started = true;
    KL_INFO("gpu", "page table ready (%s): title memory is imported as it is, no copies",
            h.caps.rebar ? "in Resizable BAR VRAM" : "in host memory");
    return true;
}

bool ensure(uint64_t address, uint64_t len) {
    if (!address) return false;
    std::lock_guard g(g_lock);
    if (!g_started) return false;
    auto it = holding(address);
    if (it != g_regions.end() && address + len <= it->second.end) return true;
    uint64_t start, end;
    if (!committed_run(address, start, end)) return false;
    // Regions inside the new run are superseded by it.
    for (auto o = g_regions.lower_bound(start); o != g_regions.end() && o->first < end;) drop(o++);
    if (auto o = holding(start); o != g_regions.end()) drop(o);
    Region r;
    bool ok = import(start, end, r);
    if (!ok) {
        // Watched (write-protected) texture pages there? Open them and try once more.
        images::open_for_import(start, end);
        ok = import(start, end, r);
    }
    if (!ok) {
        if (g_stats.failures++ < 16)
            KL_WARN("gpu", "could not import title memory %llx-%llx (%llu KiB) for the GPU", (unsigned long long)start,
                    (unsigned long long)end, (unsigned long long)((end - start) >> 10));
        return false;
    }
    g_regions.emplace(start, r);
    map_pages(start, end, r.device);
    g_stats.regions++, g_stats.bytes += end - start;
    KL_TRACE("gpu", "imported %llx-%llx (%llu KiB)", (unsigned long long)start, (unsigned long long)end,
             (unsigned long long)((end - start) >> 10));
    return address + len <= end;
}

bool locate(uint64_t address, VkBuffer& buffer, uint64_t& offset, uint64_t& left) {
    std::lock_guard g(g_lock);
    auto it = holding(address);
    if (it == g_regions.end()) return false;
    buffer = it->second.buffer;
    offset = address - it->first;
    left = it->second.end - address;
    return true;
}

void forget(uint64_t address, uint64_t len) {
    std::lock_guard g(g_lock);
    if (!g_started) return;
    uint64_t end = address + len;
    std::vector<std::pair<uint64_t, uint64_t>> gone;
    if (auto o = holding(address); o != g_regions.end()) gone.push_back({o->first, o->second.end}), drop(o);
    for (auto o = g_regions.lower_bound(address); o != g_regions.end() && o->first < end;) {
        gone.push_back({o->first, o->second.end});
        drop(o++);
    }
    for (auto [s, e] : gone) map_pages(s, e, 0);
}

uint64_t page_table() { return g_dir.device; }

Stats stats() {
    std::lock_guard g(g_lock);
    return g_stats;
}

}  // namespace kl::gpu::memory
