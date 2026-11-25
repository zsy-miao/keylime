// keylime formats: BCPack, the lossless packing of BC1-BC7 block streams
// that the console's DirectStorage unpacks. Decode only (no BC2).
// A C++ port of VIctory's decoder (GPL-3.0, crates/victory-formats/src/bcpack.rs).
#pragma once
#include <cstddef>
#include <cstdint>

namespace kl::bcpack {

// Unpacks `src` (BC format number 1, 3-7) into `dst` (a whole number of blocks).
// Returns null on success, else what went wrong.
const char* decode(unsigned bc, const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_len);

}  // namespace kl::bcpack
