// keylime: console memory (xmem.dll). Physical pages are 64 KiB slices of
// pagefile-backed sections, mapped into host placeholders.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace kl::xmem {

constexpr size_t kPage = 64 << 10;

// Copies `len` bytes into physical pages (by page number list) starting
// `offset` bytes into the first, whether or not they are mapped anywhere.
bool write_physical(const uint64_t* pages, size_t page_count, size_t offset, const void* data, size_t len);
bool write_physical_locked(const uint64_t* pages, size_t page_count, size_t offset, const void* data, size_t len);
// Called (outside xmem's lock) with each title range write_physical changed.
using WriteHook = void (*)(uint64_t address, uint64_t len);
void set_write_hook(WriteHook h);
// Called (outside xmem's lock) when a title range starts showing other pages.
void set_remap_hook(WriteHook h);
// Called just before title memory is released (GPU imports of it must go first).
void set_free_hook(WriteHook h);
bool map_alias(uint64_t to, uint64_t from, uint64_t len);

// A view of physical pages: title addresses [va, va + len) show bytes [offset, offset + len)
// of the section `section` (an identity that is never reused; `handle` maps it).
struct PhysicalSpan {
    uint64_t section, section_bytes, va, len, offset;
    void* handle;
};
// The physical view holding title address `va`; false if `va` is not physical memory.
bool physical_at(uint64_t va, PhysicalSpan& out);
// Sections whose pages were all released since the last call.
std::vector<uint64_t> sections_gone();
bool aliased(uint64_t va, uint64_t len);

// Title memory sizes for the current console: (GPU-optimal, standard).
void title_memory(size_t& gpu_optimal, size_t& standard);

// The title's general allocator (its hooks if it has them), as XMemAlloc / XMemFree.
void* alloc(size_t size, uint64_t attributes);
void free(void* p, uint64_t attributes);
// Zeroed, committed memory below 2^40 (where console GPU addresses live), aligned to 64 KiB at least.
void* gpu_alloc(size_t size, size_t alignment);
void gpu_free(void* p);

}  // namespace kl::xmem
