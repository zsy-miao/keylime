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

// Title memory sizes for the current console: (GPU-optimal, standard).
void title_memory(size_t& gpu_optimal, size_t& standard);

}  // namespace kl::xmem
