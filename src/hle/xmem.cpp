// keylime: xmem.dll. Ported from VIctory's model (sections + placeholders),
// without its write watching: keylime's GPU side tracks buffer ownership.
#include "hle/xmem.h"

#include <windows.h>
#include <psapi.h>

#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "hle/hle.h"
#include "log.h"
#include "runtime.h"

#pragma comment(lib, "onecore.lib")  // VirtualAlloc2, MapViewOfFile3, UnmapViewOfFile2

namespace kl::xmem {
namespace {

constexpr size_t MiB = 1ull << 20, GiB = 1ull << 30;
constexpr uint32_t PAGES_2M = 0x20000000, PAGES_64K_BIT = 0x400000, PAGES_64K = PAGES_2M | PAGES_64K_BIT;
constexpr uint32_t HOST_PROTECTION = 0x1ff;
constexpr uint64_t GRAPHICS = 3, ADDRESS_32BIT = 4;
constexpr size_t FRAMES = kPage >> 12;  // page numbers count 4 KiB frames

std::atomic<size_t> g_transferred{0};

// Address window a request's flags allow: low 4 GiB for 32-bit memory, else
// 4 GiB .. 2^(47-field) (40 bits without a field: the BVH builder refuses higher).
void address_range(uint64_t flags, uintptr_t& lo, uintptr_t& hi) {
    if (flags & ADDRESS_32BIT) {
        lo = 0;
        hi = 0xFFFFFFFF;
        return;
    }
    unsigned field = unsigned(flags >> 12 & 7);
    unsigned bits = field ? 47 - field : 40;
    lo = 1ull << 32;
    hi = (1ull << bits) - 1;
}

// ------------------------------------------------------------ physical ----

std::atomic<uint64_t> g_next_section{1};
std::mutex g_gone_lock;
std::vector<uint64_t> g_gone;  // sections released, for sections_gone()
struct Section {
    HANDLE h;
    uint64_t id, bytes;
    ~Section() {
        CloseHandle(h);
        std::lock_guard g(g_gone_lock);
        g_gone.push_back(id);
    }
};
struct Slice {
    std::shared_ptr<Section> section;
    size_t index = 0;
    std::pair<uintptr_t, size_t> key(size_t k) const { return {uintptr_t(section.get()), index + k}; }
};
struct PairHash {
    size_t operator()(const std::pair<uintptr_t, size_t>& p) const { return p.first * 31 ^ p.second * 0x9E3779B97F4A7C15ull; }
};
struct Physical {
    uint64_t next_page = 0x100000;  // opaque to titles; 0 stays unused
    std::unordered_map<uint64_t, Slice> pages;
    std::map<uintptr_t, std::pair<size_t, Slice>> views;  // base -> (len, first slice)
    std::unordered_map<std::pair<uintptr_t, size_t>, std::vector<uintptr_t>, PairHash> shown;
};
std::mutex g_lock;
Physical g_phys;

bool commit(uintptr_t base, size_t len) {
    if (VirtualAlloc(reinterpret_cast<void*>(base), len, MEM_COMMIT, PAGE_READWRITE)) return true;
    DWORD e = GetLastError();
    if (e == ERROR_COMMITMENT_LIMIT) {
        PERFORMANCE_INFORMATION pi{sizeof pi};
        GetPerformanceInfo(&pi, sizeof pi);
        KL_ERROR("xmem", "Windows commit limit reached committing %zu bytes (%zu of %zu MiB in use): raise the page file",
                 len, pi.CommitTotal * pi.PageSize / MiB, pi.CommitLimit * pi.PageSize / MiB);
    } else {
        KL_WARN("xmem", "committing physical pages failed: %lu", e);
    }
    return false;
}

bool map_view(Physical& p, uintptr_t base, size_t len, const Slice& slice);

void unmap(Physical& p, uintptr_t base, size_t len) {
    if (!len) return;
    uintptr_t end = base + len;
    std::vector<uintptr_t> hit;
    auto it = p.views.upper_bound(base);
    if (it != p.views.begin()) {
        auto prev = std::prev(it);
        if (prev->first + prev->second.first > base) hit.push_back(prev->first);
    }
    for (; it != p.views.end() && it->first < end; ++it) hit.push_back(it->first);
    for (uintptr_t at : hit) {
        auto node = p.views.find(at);
        if (node == p.views.end()) continue;
        auto [size, slice] = node->second;
        p.views.erase(node);
        for (size_t k = 0; k < size / kPage; ++k) {
            auto f = p.shown.find(slice.key(k));
            if (f == p.shown.end()) continue;
            std::erase(f->second, at + k * kPage);
            if (f->second.empty()) p.shown.erase(f);
        }
        UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(at), MEM_PRESERVE_PLACEHOLDER);
        if (at < base) map_view(p, at, base - at, slice);
        if (at + size > end) map_view(p, end, at + size - end, Slice{slice.section, slice.index + (end - at) / kPage});
    }
}

bool map_view(Physical& p, uintptr_t base, size_t len, const Slice& slice) {
    // Split the placeholder to exactly [base, base+len) (no-op when it already is), then map there.
    VirtualFree(reinterpret_cast<void*>(base), len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
    void* view = MapViewOfFile3(slice.section->h, GetCurrentProcess(), reinterpret_cast<void*>(base),
                                uint64_t(slice.index) * kPage, len, MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, nullptr, 0);
    if (uintptr_t(view) != base) {
        KL_WARN("xmem", "mapping physical pages at %llx (+%zu) failed: %lu (not a placeholder?)",
                (unsigned long long)base, len, GetLastError());
        return false;
    }
    if (!commit(base, len)) {
        UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
        return false;
    }
    for (size_t k = 0; k < len / kPage; ++k) p.shown[slice.key(k)].push_back(base + k * kPage);
    p.views[base] = {len, slice};
    return true;
}

// ------------------------------------------------------------- exports ----

void* WINAPI XMemVirtualAlloc(void* base, SIZE_T size, DWORD kind, uint64_t flags, DWORD protect) {
    uintptr_t lo, hi;
    address_range(flags, lo, hi);
    MEM_ADDRESS_REQUIREMENTS req{};
    req.LowestStartingAddress = reinterpret_cast<void*>(lo);
    req.HighestEndingAddress = reinterpret_cast<void*>(hi);
    req.Alignment = (kind & (PAGES_2M | PAGES_64K_BIT)) == PAGES_2M ? 2 * MiB : 0;
    DWORD prot = (protect & HOST_PROTECTION) ? (protect & HOST_PROTECTION) : PAGE_NOACCESS;
    // A reservation of 64 KiB pages takes physical pages later: only a placeholder can.
    const bool placeholder = (kind & PAGES_64K) == PAGES_64K && !(kind & MEM_COMMIT);
    if (placeholder) {
        if (kind & MEM_REPLACE_PLACEHOLDER) return base;  // part of a split placeholder: stays one
        prot = PAGE_NOACCESS;
    }
    DWORD k = kind & ~(PAGES_2M | PAGES_64K_BIT);
    if (placeholder) k |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    MEM_EXTENDED_PARAMETER param{};
    param.Type = MemExtendedParameterAddressRequirements;
    param.Pointer = &req;
    const bool constrained = base == nullptr;
    if (constrained) k |= MEM_RESERVE;
    void* at = VirtualAlloc2(GetCurrentProcess(), base, size, k, prot, constrained ? &param : nullptr, constrained ? 1 : 0);
    KL_TRACE("xmem", "XMemVirtualAlloc(%p, %zu, kind %lx, flags %llx, prot %lx) -> %p", base, size, kind,
             (unsigned long long)flags, protect, at);
    if (!at && constrained)
        KL_WARN("xmem", "XMemVirtualAlloc(%zu bytes, kind %lx, flags %llx) failed: %lu", size, kind,
                (unsigned long long)flags, GetLastError());
    return at;
}

struct BasicInformation {
    MEMORY_BASIC_INFORMATION host;
    uint64_t flags;
    uint32_t tag_code;
    uint8_t tag[4];
};

BOOL WINAPI XMemVirtualQuery(HANDLE process, const void* address, BasicInformation* out) {
    if (!VirtualQueryEx(process, address, &out->host, sizeof out->host)) return FALSE;
    out->flags = out->host.State == MEM_COMMIT ? GRAPHICS : 0;  // everything is GPU-reachable here
    out->tag_code = 0;
    std::memset(out->tag, 0, 4);
    return TRUE;
}

// XMemAlloc attributes: log2 alignment (5 bits at 24), bit 33 = 32-bit address.
constexpr uint64_t ALIGNMENT_SHIFT = 24, ALLOC_32BIT = 1ull << 33;

using AllocFn = void*(WINAPI*)(SIZE_T, uint64_t);
using FreeFn = void(WINAPI*)(void*, uint64_t);

// The title exports __XMemAllocationHooks__ {alloc, free}: XMemAlloc/XMemFree go through them.
void** hooks() {
    static void** at = [] {
        void** p = nullptr;
        if (auto* m = loader().main()) p = reinterpret_cast<void**>(GetProcAddress(m->module, "__XMemAllocationHooks__"));
        KL_INFO("xmem", "allocation hooks %s", p ? "found" : "absent");
        return p;
    }();
    return at;
}

void* WINAPI XMemAllocDefault(SIZE_T size, uint64_t attributes) {
    if (attributes & ALLOC_32BIT)
        return XMemVirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, GRAPHICS | ADDRESS_32BIT, PAGE_READWRITE);
    unsigned shift = unsigned(attributes >> ALIGNMENT_SHIFT & 0x1f);
    size_t align = size_t(1) << (shift < 4 ? 4 : shift);
    return _aligned_malloc(size ? size : 1, align);
}

void before_free(void* p);
void WINAPI XMemFreeDefault(void* address, uint64_t attributes) {
    if (!address) return;
    if (attributes & ALLOC_32BIT) {
        before_free(address);
        VirtualFree(address, 0, MEM_RELEASE);
        return;
    }
    _aligned_free(address);
}

void* WINAPI XMemAlloc(SIZE_T size, uint64_t attributes) {
    void** h = hooks();
    if (h && h[0]) return reinterpret_cast<AllocFn>(h[0])(size, attributes);
    return XMemAllocDefault(size, attributes);
}

void WINAPI XMemFree(void* address, uint64_t attributes) {
    void** h = hooks();
    if (h && h[1]) return reinterpret_cast<FreeFn>(h[1])(address, attributes);
    XMemFreeDefault(address, attributes);
}

BOOL WINAPI XMemIsToolsMemAvailable() { return FALSE; }

BOOL WINAPI XMemAllocatePhysicalPages(uint64_t flags, SIZE_T* count, uint64_t* pages) {
    if (!count || !pages || !*count) return FALSE;
    size_t n = *count, size = n * kPage;
    // Reserved, committed per mapping: the title asks for ~6 GiB at once.
    HANDLE h = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_RESERVE, DWORD(uint64_t(size) >> 32),
                                  DWORD(size), nullptr);
    if (!h) {
        KL_WARN("xmem", "XMemAllocatePhysicalPages(%zu) failed: %lu", n, GetLastError());
        *count = 0;
        return FALSE;
    }
    KL_INFO("xmem", "XMemAllocatePhysicalPages: %zu pages (%zu MiB), flags %llx", n, size / MiB, (unsigned long long)flags);
    auto section = std::shared_ptr<Section>(new Section{h, g_next_section++, uint64_t(size)});
    std::lock_guard g(g_lock);
    uint64_t first = g_phys.next_page;
    g_phys.next_page += n * FRAMES;
    for (size_t i = 0; i < n; ++i) {
        uint64_t number = first + i * FRAMES;
        g_phys.pages[number] = Slice{section, i};
        pages[i] = number;
    }
    return TRUE;
}

BOOL WINAPI XMemFreePhysicalPages(SIZE_T n, const uint64_t* pages) {
    if (!pages) return FALSE;
    std::lock_guard g(g_lock);
    for (size_t i = 0; i < n; ++i) g_phys.pages.erase(pages[i]);
    return TRUE;
}

WriteHook g_write_hook = nullptr, g_remap_hook = nullptr, g_free_hook = nullptr;
void before_free(void* p) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (g_free_hook && p && VirtualQuery(p, &mbi, sizeof mbi) && mbi.AllocationBase == p) {
        // The whole allocation: walk its regions.
        uintptr_t end = uintptr_t(p);
        while (VirtualQuery(reinterpret_cast<void*>(end), &mbi, sizeof mbi) && mbi.AllocationBase == p) end += mbi.RegionSize;
        g_free_hook(uint64_t(uintptr_t(p)), uint64_t(end - uintptr_t(p)));
    }
}

void* map_physical_locked(void* address, SIZE_T n, const uint64_t* pages);
void* WINAPI XMemMapPhysicalPages(void* address, SIZE_T n, const uint64_t* pages) {
    void* r;
    {
        std::lock_guard g(g_lock);
        r = map_physical_locked(address, n, pages);
    }
    // The range shows other pages now: the GPU's view of it and copies made from it are stale.
    if (g_remap_hook && address && n) g_remap_hook(uint64_t(address), uint64_t(n) * kPage);
    return r;
}

void* map_physical_locked(void* address, SIZE_T n, const uint64_t* pages) {
    uintptr_t base = uintptr_t(address);
    unmap(g_phys, base, n * kPage);
    if (!pages) return address;
    for (size_t i = 0; i < n;) {
        auto f = g_phys.pages.find(pages[i]);
        if (f == g_phys.pages.end()) {
            KL_WARN("xmem", "XMemMapPhysicalPages: unknown page %llx", (unsigned long long)pages[i]);
            return nullptr;
        }
        const Slice first = f->second;
        size_t run = 1;
        while (i + run < n) {
            auto s = g_phys.pages.find(pages[i + run]);
            if (s == g_phys.pages.end() || s->second.section != first.section || s->second.index != first.index + run) break;
            ++run;
        }
        if (!map_view(g_phys, base + i * kPage, run * kPage, first)) return nullptr;
        i += run;
    }
    return address;
}

BOOL WINAPI XMemMapPhysicalPagesScatter(void* const* addresses, SIZE_T n, const uint64_t* pages) {
    if (!addresses) return FALSE;
    for (size_t i = 0; i < n; ++i) {
        const uint64_t* page = (!pages || !pages[i]) ? nullptr : &pages[i];
        if (!XMemMapPhysicalPages(addresses[i], 1, page)) return FALSE;
    }
    return TRUE;
}

struct WorkingSet {
    size_t title_limit, title_used, tools_limit, tools_used;
    uint32_t stack_used, physical_pages;
    size_t gpu_optimal_limit, gpu_optimal_used, tools_gpu_optimal_limit, tools_gpu_optimal_used;
    uint32_t kernel_pool_used, kernel_pool_limit;
};

BOOL WINAPI XMemGetWorkingSetStatistics(uint32_t, WorkingSet* out) {
    if (!out) return FALSE;
    PROCESS_MEMORY_COUNTERS pmc{sizeof pmc};
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc);
    size_t gpu, standard;
    title_memory(gpu, standard);
    *out = {};
    out->title_limit = gpu + standard - g_transferred.load();
    out->title_used = pmc.PagefileUsage;
    out->gpu_optimal_limit = gpu;
    return TRUE;
}

BOOL WINAPI XMemTransferMemory(uint32_t kind, SIZE_T size) {
    KL_INFO("xmem", "XMemTransferMemory(%u, %zu)", kind, size);
    if (!size) return TRUE;
    size = (size + 2 * MiB - 1) & ~(2 * MiB - 1);
    size_t gpu, standard;
    title_memory(gpu, standard);
    if (kind != 0) return FALSE;  // only title -> system; no tools memory on retail
    size_t cur = g_transferred.load();
    do {
        if (cur + size > standard) return FALSE;
    } while (!g_transferred.compare_exchange_weak(cur, cur + size));
    return TRUE;
}

}  // namespace

void title_memory(size_t& gpu_optimal, size_t& standard) {
    if (runtime().console == Console::SeriesS) {
        gpu_optimal = 0;
        standard = size_t(7776) << 20;
    } else {
        gpu_optimal = 10 * GiB;
        standard = 7 * GiB / 2;
    }
}

void* alloc(size_t size, uint64_t attributes) { return XMemAlloc(size, attributes); }
void free(void* p, uint64_t attributes) { XMemFree(p, attributes); }

void* gpu_alloc(size_t size, size_t alignment) {
    MEM_ADDRESS_REQUIREMENTS req{};
    req.LowestStartingAddress = reinterpret_cast<void*>(1ull << 32);
    req.HighestEndingAddress = reinterpret_cast<void*>((1ull << 40) - 1);
    req.Alignment = alignment > 65536 ? alignment : 0;
    MEM_EXTENDED_PARAMETER param{};
    param.Type = MemExtendedParameterAddressRequirements;
    param.Pointer = &req;
    return VirtualAlloc2(GetCurrentProcess(), nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE, &param, 1);
}
void gpu_free(void* p) {
    before_free(p);
    if (p) VirtualFree(p, 0, MEM_RELEASE);
}

void set_remap_hook(WriteHook h) { g_remap_hook = h; }
void set_free_hook(WriteHook h) { g_free_hook = h; }

// True if any page of [va, va + len) is physical memory shown at another address too
// (writes there change this range without touching it).
bool aliased(uint64_t va, uint64_t len) {
    std::lock_guard g(g_lock);
    for (uint64_t a = va & ~uint64_t(kPage - 1); a < va + len; a += kPage) {
        auto it = g_phys.views.upper_bound(uintptr_t(a));
        if (it == g_phys.views.begin()) continue;
        --it;
        if (a >= it->first + it->second.first) continue;
        Slice s{it->second.second.section, it->second.second.index + (a - it->first) / kPage};
        auto sh = g_phys.shown.find(s.key(0));
        if (sh != g_phys.shown.end() && sh->second.size() > 1) return true;
    }
    return false;
}

// Shows the physical pages mapped at [from, from + len) at `to` as well (in place of what `to`
// showed); `from` 0 unmaps. The GPU queue's page-mapping copies: titles stream into pool pages and
// map them into a tiled heap this way. False if a source page is not mapped physical memory.
bool map_alias(uint64_t to, uint64_t from, uint64_t len) {
    bool ok = true;
    {
        std::lock_guard g(g_lock);
        unmap(g_phys, uintptr_t(to), size_t(len));
        for (uint64_t k = 0; from && k < len; k += kPage) {
            uintptr_t at = uintptr_t(from + k);
            auto it = g_phys.views.upper_bound(at);
            if (it == g_phys.views.begin()) { ok = false; break; }
            --it;
            if (at >= it->first + it->second.first) { ok = false; break; }
            Slice s{it->second.second.section, it->second.second.index + (at - it->first) / kPage};
            if (!map_view(g_phys, uintptr_t(to + k), kPage, s)) { ok = false; break; }
        }
    }
    if (g_remap_hook) g_remap_hook(to, len);  // new contents there (outside our lock)
    return ok;
}
thread_local std::vector<std::pair<uint64_t, uint64_t>> g_notes;

bool physical_at(uint64_t va, PhysicalSpan& out) {
    std::lock_guard g(g_lock);
    auto it = g_phys.views.upper_bound(uintptr_t(va));
    if (it == g_phys.views.begin()) return false;
    --it;
    uint64_t base = it->first, len = it->second.first;
    if (va >= base + len) return false;
    const Slice& s = it->second.second;
    out.section = s.section->id;
    out.handle = s.section->h;
    out.section_bytes = s.section->bytes;
    out.va = base;
    out.len = len;
    out.offset = uint64_t(s.index) * kPage;
    return true;
}

std::vector<uint64_t> sections_gone() {
    std::lock_guard g(g_gone_lock);
    std::vector<uint64_t> v;
    v.swap(g_gone);
    return v;
}

void set_write_hook(WriteHook h) { g_write_hook = h; }

bool write_physical(const uint64_t* pages, size_t page_count, size_t offset, const void* data, size_t len) {
    g_notes.clear();
    bool ok = write_physical_locked(pages, page_count, offset, data, len);
    if (g_write_hook)
        for (auto [va, n] : g_notes) g_write_hook(va, n);  // outside our lock
    return ok;
}

bool write_physical_locked(const uint64_t* pages, size_t page_count, size_t offset, const void* data, size_t len) {
    const auto* src = static_cast<const uint8_t*>(data);
    std::lock_guard g(g_lock);
    size_t done = 0;
    while (done < len) {
        size_t at = offset + done, pi = at / kPage, within = at % kPage;
        if (pi >= page_count) return false;
        auto f = g_phys.pages.find(pages[pi]);
        if (f == g_phys.pages.end()) {
            static std::atomic<int> n{0};
            if (n++ < 4) KL_WARN("xmem", "a write to physical page %llx (entry %zu of %zu) fails: no allocation holds that page",
                                 (unsigned long long)pages[pi], pi, page_count);
            return false;
        }
        // one view per run of consecutive slices
        const Slice first = f->second;
        size_t run = 1;
        while (pi + run < page_count && (run * kPage - within) < (len - done)) {
            auto s = g_phys.pages.find(pages[pi + run]);
            if (s == g_phys.pages.end() || s->second.section != first.section || s->second.index != first.index + run) break;
            ++run;
        }
        uint64_t off = uint64_t(first.index) * kPage;
        void* v = MapViewOfFile(first.section->h, FILE_MAP_WRITE, DWORD(off >> 32), DWORD(off), run * kPage);
        if (!v || !VirtualAlloc(v, run * kPage, MEM_COMMIT, PAGE_READWRITE)) {
            if (v) UnmapViewOfFile(v);
            return false;
        }
        size_t k = run * kPage - within;
        if (k > len - done) k = len - done;
        std::memcpy(static_cast<uint8_t*>(v) + within, src + done, k);
        UnmapViewOfFile(v);
        // Tell whoever keeps copies of this memory (the GPU's textures) which title addresses changed.
        if (g_write_hook)
            for (size_t r = 0; r < run; ++r) {
                auto sh = g_phys.shown.find(first.key(r));
                if (sh != g_phys.shown.end())
                    for (uintptr_t va : sh->second) g_notes.push_back({va, kPage});
            }
        done += k;
    }
    return true;
}

KL_MODULE("xmem", (std::vector<std::string>{"xmem.dll"}),
          (std::vector<hle::Export>{
              KL_EXPORT(XMemAlloc), KL_EXPORT(XMemFree), KL_EXPORT(XMemAllocDefault), KL_EXPORT(XMemFreeDefault),
              KL_EXPORT(XMemIsToolsMemAvailable), KL_EXPORT(XMemAllocatePhysicalPages), KL_EXPORT(XMemFreePhysicalPages),
              KL_EXPORT(XMemMapPhysicalPages), KL_EXPORT(XMemMapPhysicalPagesScatter), KL_EXPORT(XMemVirtualAlloc),
              KL_EXPORT(XMemVirtualQuery), KL_EXPORT(XMemGetWorkingSetStatistics), KL_EXPORT(XMemTransferMemory),
          }));

}  // namespace kl::xmem
