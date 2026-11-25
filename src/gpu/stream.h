// keylime GPU: a list's recorded command stream. Title calls append compact
// commands here (CPU only, cheap). At submission the commands that only
// touch memory (value writes, buffer copies, query resolves) are carried out
// on the CPU, so the title sees its GPU work complete; the rest waits for the
// Vulkan translator.
#pragma once
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace kl::gpu {

enum class Op : uint16_t {
    SetPipeline, Draw, DrawIndexed, Dispatch, DispatchMesh, Barrier, CopyBuffer, CopyTexture, CopyResource,
    ClearRtv, ClearDsv, ClearUavUint, ClearUavFloat, Topology, Viewports, Scissors, BlendFactor, StencilRef,
    Heaps, RootSignature, RootTable, RootConstant, RootConstants, RootView, IndexBuffer, VertexBuffers, Targets,
    ExecuteIndirect, Write32, Write64, WaitValue, CopyMem, Resolve, ResolveQueries, Query, Discard, DepthBounds, Count
};

// Payloads (shared by the recorders and the executor). `gfx` is 1 for graphics, 0 for compute.
namespace cmd {
struct Draw { uint32_t count, instances, first, first_instance; };
struct DrawIndexed { uint32_t count, instances, first; int32_t base; uint32_t first_instance; };
struct Dispatch { uint32_t x, y, z, flags; };
struct CopyBuffer { uint64_t dst, src, len; };
struct Write { uint64_t address, value; uint32_t bytes, flags; };
struct ResolveQueries { uint64_t dst; uint32_t count; };
struct Root { uint32_t gfx, slot; uint64_t value; };              // table handle / view address / root signature
struct RootConstants { uint32_t gfx, slot, first, count; uint32_t values[64]; };
struct IndexBuffer { uint64_t address; uint32_t size, format; };
struct VertexView { uint64_t address; uint32_t size, stride; };
struct VertexBuffers { uint32_t start, count; VertexView views[32]; };
struct Targets { uint32_t count, single; uint64_t rtv[8]; uint64_t dsv; };
struct ClearRtv { uint64_t handle; float color[4]; };
struct ClearDsv { uint64_t handle; uint32_t flags; float depth; uint32_t stencil; };
struct ClearUav { uint64_t gpu_handle, cpu_handle, resource; uint32_t values[4]; };
// A D3D12_TEXTURE_COPY_LOCATION: resource, type (0 subresource, 1 placed footprint), then
// subresource index or {offset, format, width, height, depth, row pitch}.
struct CopyLocation {
    void* resource;
    uint32_t type, pad;
    uint64_t offset;
    uint32_t format, width, height, depth, row_pitch, pad2;
};
static_assert(sizeof(CopyLocation) == 48);
struct CopyTexture { CopyLocation dst, src; uint32_t x, y, z, has_box; uint32_t box[6]; };
struct CopyResource { void* dst; void* src; };
// ExecuteIndirect: argument and count buffers as title addresses (count 0: none).
struct Indirect { void* signature; uint64_t args, offset, max, count, count_offset; };
struct WaitValue { uint64_t address, compare, reference; uint32_t bytes, mask; };
struct Heaps { uint32_t count, pad; void* heaps[8]; };
struct Viewports { uint32_t count, pad; float v[16][6]; };
struct DepthBounds { float min, max; };
struct Scissors { uint32_t count, pad; int32_t r[16][4]; };
}  // namespace cmd

class Stream {
public:
    void open(uint32_t kind) {
        kind_ = kind, bytes_.clear(), closed_ = false, commands_ = 0;
        for (auto& c : chunks_) c.used = 0;
    }
    void close() { closed_ = true; }
    bool closed() const { return closed_; }
    uint32_t kind() const { return kind_; }
    size_t commands() const { return commands_; }

    void add(Op op, const void* payload, uint32_t size);
    template <class T>
    void add(Op op, const T& v) { add(op, &v, uint32_t(sizeof v)); }
    void set_pipeline(void* p) { add(Op::SetPipeline, p); }

    // Memory the list owns until reset (CPU writes, GPU reads at the same address).
    void* upload(uint64_t size, uint64_t alignment);

    // Walks the commands: f(op, payload, size).
    template <class F>
    void each(F&& f) const {
        for (size_t at = 0; at + 8 <= bytes_.size();) {
            uint32_t head[2];
            memcpy(head, bytes_.data() + at, 8);
            f(Op(head[0]), bytes_.data() + at + 8, head[1]);
            at += 8 + ((head[1] + 7) & ~7u);
        }
    }

private:
    std::vector<uint8_t> bytes_;
    struct Chunk { uint8_t* p; uint64_t size, used; };
    std::vector<Chunk> chunks_;  // GPU-visible memory, kept across resets
    uint32_t kind_ = 0;
    size_t commands_ = 0;
    bool closed_ = false;
};

namespace list {
Stream* stream_of(void* list);
}
namespace stream {
// The number of commands of each kind recorded (diagnostics).
// Fills the list table slots of every recorded command (older slot numbers; `newer` shifts them).
void fill_list_methods(void** table, bool newer);
// Carries out the commands that only touch memory, at submission.
void run_cpu_side(const Stream& s);
}  // namespace stream

}  // namespace kl::gpu
