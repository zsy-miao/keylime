// keylime GPU: a list's recorded command stream. Title calls append compact
// commands here (CPU only, cheap); the queue's submission thread translates a
// closed stream into Vulkan.
#pragma once
#include <cstdint>
#include <vector>

namespace kl::gpu {

enum class Op : uint16_t {
    SetPipeline, Draw, DrawIndexed, Dispatch, DispatchMesh, Barrier, CopyBuffer, CopyTexture, CopyResource,
    ClearRtv, ClearDsv, ClearUavUint, ClearUavFloat, Topology, Viewports, Scissors, BlendFactor, StencilRef,
    Heaps, RootTable, RootConstant, RootConstants, RootView, IndexBuffer, VertexBuffers, Targets,
    ExecuteIndirect, Write32, Write64, WaitValue, CopyMemory, Resolve, Query, Count
};

class Stream {
public:
    void open(uint32_t kind) { kind_ = kind, bytes_.clear(), closed_ = false, commands_ = 0; }
    void close() { closed_ = true; }
    bool closed() const { return closed_; }
    uint32_t kind() const { return kind_; }
    size_t commands() const { return commands_; }

    // Appends op + payload (copied).
    void add(Op op, const void* payload, uint32_t size);
    template <class T>
    void add(Op op, const T& v) { add(op, &v, uint32_t(sizeof v)); }
    void set_pipeline(void* p) { add(Op::SetPipeline, p); }

    const std::vector<uint8_t>& bytes() const { return bytes_; }

private:
    std::vector<uint8_t> bytes_;
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
}

}  // namespace kl::gpu
