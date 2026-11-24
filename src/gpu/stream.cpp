#include "gpu/stream.h"

#include <cstring>

namespace kl::gpu {

void Stream::add(Op op, const void* payload, uint32_t size) {
    size_t at = bytes_.size();
    uint32_t padded = (size + 7) & ~7u;
    bytes_.resize(at + 8 + padded);
    uint32_t head[2] = {uint32_t(op), size};
    std::memcpy(bytes_.data() + at, head, 8);
    if (size) std::memcpy(bytes_.data() + at + 8, payload, size);
    ++commands_;
}

namespace stream {
void fill_list_methods(void**, bool) {}
}  // namespace stream

}  // namespace kl::gpu
