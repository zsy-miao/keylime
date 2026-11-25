// keylime: console memory (xmem.dll). Physical pages are 64 KiB slices of
// pagefile-backed sections, mapped into host placeholders.
#pragma once
#include <cstddef>
#include <cstdint>

namespace kl::xmem {

constexpr size_t kPage = 64 << 10;

// Copies `len` bytes into physical pages (by page number list) starting
// `offset` bytes into the first, whether or not they are mapped anywhere.
bool write_physical(const uint64_t* pages, size_t page_count, size_t offset, const void* data, size_t len);
bool write_physical_locked(const uint64_t* pages, size_t page_count, size_t offset, const void* data, size_t len);
// Called (outside xmem's lock) with each title range write_physical changed.
using WriteHook = void (*)(uint64_t address, uint64_t len);
void set_write_hook(WriteHook h);

// Title memory sizes for the current console: (GPU-optimal, standard).
void title_memory(size_t& gpu_optimal, size_t& standard);

// The title's general allocator (its hooks if it has them), as XMemAlloc / XMemFree.
void* alloc(size_t size, uint64_t attributes);
void free(void* p, uint64_t attributes);
// Zeroed, committed memory below 2^40 (where console GPU addresses live), aligned to 64 KiB at least.
void* gpu_alloc(size_t size, size_t alignment);
void gpu_free(void* p);

}  // namespace kl::xmem
