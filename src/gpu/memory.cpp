// keylime GPU: title memory imported into Vulkan, and the GPU page table.
#include "gpu/memory.h"

#include <windows.h>

#include <cstdio>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gpu/host.h"
#include "gpu/images.h"
#include "gpu/retire.h"
#include "hle/xmem.h"
#include "log.h"

namespace kl::gpu::memory {
namespace {

constexpr uint64_t PAGE = 64 << 10, GIB = 1ull << 30;
constexpr uint64_t DIRECTORY = 1ull << 17;     // 2^47 bytes of address space / 1 GiB
constexpr uint64_t TABLE = GIB / PAGE;          // 16384 entries per 1 GiB table
constexpr uint64_t TABLE_BYTES = TABLE * 8;     // 128 KiB
constexpr uint64_t POOL_TABLES = 256;           // tables per pool block (32 MiB)
constexpr uint64_t GUARD = 1ull << 63;          // page-table entry bit: no GPU allocation covers the page
constexpr uint64_t NEXT = 1ull << 62;           // page-table entry bit: the next page follows on the device (one load may span both)
constexpr uint64_t GUARD_RECORD = DIRECTORY * 8; // in the directory block: u32 count, u32 shader, u64 address

struct Region {
    uint64_t end;
    VkBuffer buffer;
    VkDeviceMemory memory;
    uint64_t device;
    bool owned = true;    // false: a window on a shadow (which owns the buffer)
    uint64_t offset = 0;  // where `start` is in `buffer`
};

// Physical memory (xmem sections) is imported once per SHADOW bytes of each section, through
// our own view of it: every title address showing those pages is a window on that import.
// (Importing each title mapping instead costs one Vulkan allocation per 64 KiB alias, and the
// driver runs out of allocations long before memory.)
constexpr uint64_t SHADOW = GIB;
struct Shadow {
    void* view = nullptr;
    uint64_t len = 0;
    Region r{};
};
std::map<std::pair<uint64_t, uint64_t>, Shadow> g_shadows;  // (section, chunk)

std::mutex g_lock;
std::map<uint64_t, Region> g_regions;  // by start
Block g_dir, g_scratch, g_sink;  // sink: where shader stores that must not land go (no branch in the shader)
std::vector<Block> g_pools;
uint32_t g_pool_used = POOL_TABLES;  // tables used in the last pool
uint64_t* g_tables[DIRECTORY] = {};   // host view of each directory entry's table (null: the empty one)
uint64_t g_empty_table_device = 0;
uint64_t* g_empty_table = nullptr;
Stats g_stats{};
bool g_started = false;
bool g_guard = true;
std::unordered_set<uint64_t> g_owned;      // pages a GPU allocation covers
std::map<uint64_t, uint64_t> g_allocs;      // start -> length of each
std::unordered_map<uint32_t, std::pair<uint64_t, int>> g_shader_names;

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
        uint64_t e = device ? device + (p - start) : g_scratch.device;  // wraps for a partial first page
        if (device && g_guard && !g_owned.count(p >> 16)) e |= GUARD;
        t[(p >> 16) & (TABLE - 1)] = e;
    }
    // NEXT bits of the pages around the range: set where the following page continues on the device.
    auto slot = [](uint64_t p) -> uint64_t* {
        uint64_t d = p >> 30;
        if (d >= DIRECTORY || !g_tables[d]) return nullptr;
        return &g_tables[d][(p >> 16) & (TABLE - 1)];
    };
    uint64_t first = (start & ~(PAGE - 1)) - PAGE;
    for (uint64_t p = first; p < end; p += PAGE) {
        uint64_t* cur = slot(p);
        if (!cur) continue;
        uint64_t* nxt = slot(p + PAGE);
        uint64_t a = *cur & ~(GUARD | NEXT), z = nxt ? *nxt & ~(GUARD | NEXT) : 0;
        bool cont = nxt && a != g_scratch.device && z == a + PAGE && ((*cur ^ *nxt) & GUARD) == 0;
        *cur = cont ? *cur | NEXT : *cur & ~NEXT;
    }
}

void drop(std::map<uint64_t, Region>::iterator it) {
    Region r = it->second;
    uint64_t start = it->first;
    g_stats.regions--, g_stats.bytes -= r.end - start;
    g_regions.erase(it);
    if (!r.owned) return;
    g_stats.imports--;
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
    g_stats.imports++;
    return true;
}

std::map<uint64_t, Region>::iterator holding(uint64_t a);

// Shadows of sections that are gone: their imports go (no title address shows them any more).
void sweep_shadows() {
    for (uint64_t id : xmem::sections_gone())
        for (auto it = g_shadows.lower_bound({id, 0}); it != g_shadows.end() && it->first.first == id;) {
            Shadow sh = it->second;
            it = g_shadows.erase(it);
            g_stats.imports--, g_stats.shadow_bytes -= sh.len;
            retire([sh] {
                auto dev = host::get().device;
                vk::vkDestroyBuffer(dev, sh.r.buffer, nullptr);
                vk::vkFreeMemory(dev, sh.r.memory, nullptr);
                UnmapViewOfFile(sh.view);
            });
        }
}

// The import of chunk `chunk` of a section (made on first use).
Shadow* shadow_of(const xmem::PhysicalSpan& p, uint64_t chunk) {
    auto key = std::make_pair(p.section, chunk);
    if (auto it = g_shadows.find(key); it != g_shadows.end()) return &it->second;
    uint64_t off = chunk * SHADOW, len = std::min(SHADOW, p.section_bytes - off);
    void* v = MapViewOfFile(HANDLE(p.handle), FILE_MAP_WRITE, DWORD(off >> 32), DWORD(off), size_t(len));
    // The section is reserved, committed as used: commit the whole chunk so it can be imported.
    if (!v || !VirtualAlloc(v, size_t(len), MEM_COMMIT, PAGE_READWRITE)) {
        if (v) UnmapViewOfFile(v);
        if (g_stats.failures++ < 16) KL_WARN("gpu", "could not map %llu MiB of physical memory for the GPU: %lu",
                                             (unsigned long long)(len >> 20), GetLastError());
        return nullptr;
    }
    Shadow sh;
    sh.view = v, sh.len = len;
    if (!import(uint64_t(v), uint64_t(v) + len, sh.r)) {
        UnmapViewOfFile(v);
        if (g_stats.failures++ < 16) KL_WARN("gpu", "could not import %llu MiB of physical memory for the GPU",
                                             (unsigned long long)(len >> 20));
        return nullptr;
    }
    g_stats.shadow_bytes += len;
    KL_INFO("gpu", "physical memory: section %llu bytes %llx+%llx imported once (%llu MiB of shadows)",
            (unsigned long long)p.section, (unsigned long long)off, (unsigned long long)len,
            (unsigned long long)(g_stats.shadow_bytes >> 20));
    return &g_shadows.emplace(key, sh).first->second;
}

// Title address `address` is physical memory: make the part of its view inside one shadow
// chunk a window on that shadow. 0: not physical memory; -1: physical but failed; 1: done.
int ensure_physical(uint64_t address, uint64_t len) {
    xmem::PhysicalSpan p;
    if (!xmem::physical_at(address, p)) return 0;
    uint64_t at = p.offset + (address - p.va);  // section offset of `address`
    uint64_t chunk = at / SHADOW;
    Shadow* sh = shadow_of(p, chunk);
    if (!sh) return -1;
    uint64_t chunk_start = chunk * SHADOW, chunk_end = chunk_start + sh->len;
    // The window: the view's addresses whose section bytes fall in this chunk.
    uint64_t lo = std::max(p.offset, chunk_start), hi = std::min(p.offset + p.len, chunk_end);
    uint64_t start = p.va + (lo - p.offset), end = p.va + (hi - p.offset);
    for (auto o = g_regions.lower_bound(start); o != g_regions.end() && o->first < end;) drop(o++);
    if (auto o = holding(start); o != g_regions.end()) drop(o);
    Region r = sh->r;
    r.end = end;
    r.device = sh->r.device + (lo - chunk_start);
    r.owned = false;
    r.offset = lo - chunk_start;
    g_regions.emplace(start, r);
    map_pages(start, end, r.device);
    g_stats.regions++, g_stats.bytes += end - start;
    return address + len <= end ? 1 : -1;
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

void describe(uint64_t a, char* out, size_t n) {
    auto it = g_allocs.upper_bound(a);
    if (it == g_allocs.begin()) {
        snprintf(out, n, "no GPU allocation below it");
        return;
    }
    --it;
    snprintf(out, n, "%llu bytes past the end of the GPU allocation at %llx (%llu bytes)",
             (unsigned long long)(a - it->first - it->second), (unsigned long long)it->first, (unsigned long long)it->second);
}

// Shaders count the stores they skipped on guarded pages and keep the first one's address.
void watch_guard() {
    SetThreadDescription(GetCurrentThread(), L"keylime-store-guard");
    auto* rec = reinterpret_cast<volatile uint32_t*>(g_dir.map + GUARD_RECORD);
    std::unordered_set<uint64_t> told;
    uint64_t total = 0;
    for (;;) {
        Sleep(2);
        uint32_t n = rec[0];
        if (!n) continue;
        uint32_t shader = rec[1];
        uint64_t a = rec[2] | uint64_t(rec[3]) << 32;
        rec[0] = 0;
        total += n;
        if (told.size() >= 32 || !told.insert(a >> 16).second) continue;
        char where[160], who[64] = "";
        {
            std::lock_guard g(g_lock);
            describe(a, where, sizeof where);
            if (auto it = g_shader_names.find(shader); it != g_shader_names.end())
                snprintf(who, sizeof who, " (%s %016llx)", it->second.second == 1 ? "pixel shader" : it->second.second == 2 ? "compute shader" : "shader",
                         (unsigned long long)it->second.first);
        }
        KL_WARN("gpu", "shader %08x%s stored to %llx, where no GPU resource or heap lies: skipped (%s; %llu such stores so far)", shader, who,
                (unsigned long long)a, where, (unsigned long long)total);
    }
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
    if (!make_block(2 * PAGE, true, g_scratch) || !make_block(GUARD_RECORD + 64, true, g_dir) || !make_block(PAGE, false, g_sink)) return false;
    memset(g_scratch.map, 0, size_t(g_scratch.size));
    memset(g_dir.map + GUARD_RECORD, 0, 64);
    memcpy(g_dir.map + GUARD_RECORD + 16, &g_sink.device, 8);     // shaders read the sink's address here
    memcpy(g_dir.map + GUARD_RECORD + 24, &g_scratch.device, 8);  // and a zero page's
    char v[4];
    g_guard = !(GetEnvironmentVariableA("KEYLIME_STORE_GUARD", v, sizeof v) && v[0] == '0');
    if (g_guard) std::thread(watch_guard).detach();
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
    sweep_shadows();
    if (int r = ensure_physical(address, len)) return r > 0;
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
    offset = it->second.offset + (address - it->first);
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

void remapped(uint64_t address, uint64_t len) {
    std::lock_guard g(g_lock);
    if (!g_started) return;
    uint64_t end = address + len;
    // What the GPU saw there goes...
    std::vector<std::pair<uint64_t, uint64_t>> gone;
    if (auto o = holding(address); o != g_regions.end()) gone.push_back({o->first, o->second.end}), drop(o);
    for (auto o = g_regions.lower_bound(address); o != g_regions.end() && o->first < end;) {
        gone.push_back({o->first, o->second.end});
        drop(o++);
    }
    for (auto [s, e] : gone) map_pages(s, e, 0);
    // ...and physical pages shown there now are windows on their shadows at once (resources made
    // earlier on this range keep working; nothing else would map it again).
    sweep_shadows();
    for (uint64_t a = address & ~(PAGE - 1); a < end;) {
        xmem::PhysicalSpan p;
        if (!xmem::physical_at(a, p)) {
            a += PAGE;
            continue;
        }
        ensure_physical(a, 1);
        auto it = holding(a);
        a = it != g_regions.end() ? std::max(it->second.end, a + PAGE) : a + PAGE;
    }
}

uint64_t page_table() { return g_dir.device; }

void own(uint64_t address, uint64_t len, bool allocation) {
    if (!address || !len || !g_guard) return;
    std::lock_guard g(g_lock);
    if (allocation) {
        auto& l = g_allocs[address];
        l = std::max(l, len);
    }
    for (uint64_t p = address >> 16; p <= (address + len - 1) >> 16; ++p) {
        if (!g_owned.insert(p).second) continue;
        uint64_t d = p >> 14;
        if (d < DIRECTORY && g_tables[d]) g_tables[d][p & (TABLE - 1)] &= ~GUARD;
    }
}

void name_shader(uint32_t id, uint64_t hash, int stage) {
    std::lock_guard g(g_lock);
    g_shader_names[id] = {hash, stage};
}

Stats stats() {
    std::lock_guard g(g_lock);
    return g_stats;
}

}  // namespace kl::gpu::memory
