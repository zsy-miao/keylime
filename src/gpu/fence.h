// keylime GPU: fences. Each is a Vulkan timeline semaphore plus the 64-bit
// value the title reads straight from memory (its "GPU address"); a
// completion thread mirrors semaphore progress into that value and sets
// the events waiting on it. Queue signals/waits stay on the GPU.
#pragma once
#include <cstdint>

#include "gpu/object.h"
#include "gpu/vk.h"

namespace kl::gpu::fence {

extern const Iid PAGEABLE_IIDS[3];
struct Fence;
Fence* from(void* obj);  // null if `obj` is not one of our fences
VkSemaphore semaphore(Fence* f);
// A queue submitted a signal of `value`: watch for it on the completion thread.
void watch(Fence* f, uint64_t value);
// CPU-side signal (also what a queue signal amounts to once reached).
void signal_cpu(Fence* f, uint64_t value);
uint64_t completed(Fence* f);
bool raise(Fence* f, uint64_t value);
uint64_t high(Fence* f);
void dump_state();

HRESULT WINAPI create(void* device, uint64_t initial, uint32_t flags, const Iid* iid, void** out);

}  // namespace kl::gpu::fence
