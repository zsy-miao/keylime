// keylime GPU: running title command lists on the host GPU.
//
// At submission each list's recorded stream becomes one Vulkan command
// buffer: pipelines are built from translated shaders by background workers
// from the moment the title creates them, root arguments become a small per-draw record of user SGPRs that
// shaders read through a push-constant pointer, descriptors stay the title's
// own RDNA 2 words in title memory, and every image lives in GENERAL layout
// with full barriers where the title asks for any.
#pragma once
#include <cstdint>
#include <functional>

#include "gpu/vk.h"

namespace kl::gpu::pipeline {
struct State;
}

namespace kl::gpu::exec {

bool start();  // once, after images::start
// A pipeline state object was created: its shaders start translating and compiling now.
void prepare(pipeline::State* st);
// The title recorded a list that sets this pipeline: a compute pipeline starts its quick build
// now, so the dispatches that use it find it ready instead of waiting for builds one by one.
void wanted(void* pipeline_object);
int precompile(const char* dir);  // keylime precompile <dir>

constexpr int GRAPHICS = 0, COMPUTE = 1;
int queue_for(uint32_t list_kind);  // D3D12 list type -> our queue
VkQueue vk_queue(int q);

void execute(int q, uint32_t count, void* const* lists);
void signal(int q, VkSemaphore timeline, uint64_t value);
void wait(int q, VkSemaphore timeline, uint64_t value, bool signal_submitted);
void dump_state();
// A semaphore was signalled from the host: held submissions that wait for it go to the driver.
void kick();
// Writes cache\used-pipelines.txt now (and the driver's pipeline cache with `with_cache`).
void save_lists(bool with_cache);
[[noreturn]] void device_lost(const char* where);
// Records `f` into a command buffer and submits it on `q`, waiting for / signalling binary
// semaphores (either may be null). Used by presentation. Returns false if nothing was submitted.
bool run(int q, const std::function<void(VkCommandBuffer)>& f, VkSemaphore wait_binary, VkSemaphore signal_binary);
// Calls `f` with the queue's lock held (for vkQueuePresentKHR).
void with_queue(int q, const std::function<void(VkQueue)>& f);

}  // namespace kl::gpu::exec
