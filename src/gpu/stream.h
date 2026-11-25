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
    ExecuteIndirect, Write32, Write64, WaitValue, CopyMem, Resolve, ResolveQueries, Query, Discard, Count
};

class Stream {
public:
    void open(uint32_t kind) {
        kind_ = kind, bytes_.clear(), closed_ = false, commands_ = 0;
        uploads_.clear();
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
    std::vector<std::unique_ptr<uint8_t[]>> uploads_;
    uint32_t kind_ = 0;
    size_t commands_ = 0;
    bool closed_ = false;
};

namespace list {
Stream* stream_of(void* list);
}
namespace stream {
// Fills the list table slots of every recorded command (older slot numbers; `newer` shifts them).
void fill_list_methods(void** table, bool newer);
// Carries out the commands that only touch memory, at submission.
void run_cpu_side(const Stream& s);
}  // namespace stream

}  // namespace kl::gpu
