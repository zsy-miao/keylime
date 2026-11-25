// keylime GPU: list command recorders. Slot map and argument shapes after
// VIctory (list.rs); recording + CPU-side execution are keylime's own.
#include "gpu/stream.h"

#include <windows.h>

#include <atomic>
#include <chrono>
#include <cstring>

#include "gpu/memory.h"
#include "gpu/resource.h"
#include "log.h"
#include "stats.h"

namespace kl::gpu {

void Stream::add(Op op, const void* payload, uint32_t size) {
    size_t at = bytes_.size();
    uint32_t padded = (size + 7) & ~7u;
    bytes_.resize(at + 8 + padded);
    uint32_t head[2] = {uint32_t(op), size};
    memcpy(bytes_.data() + at, head, 8);
    if (size) memcpy(bytes_.data() + at + 8, payload, size);
    ++commands_;
}

void* Stream::upload(uint64_t size, uint64_t alignment) {
    if (alignment < 16) alignment = 16;
    if (alignment & (alignment - 1)) return nullptr;
    for (auto& c : chunks_) {
        uint64_t at = (c.used + alignment - 1) & ~(alignment - 1);
        if (at + size <= c.size) {
            c.used = at + size;
            return c.p + at;
        }
    }
    uint64_t n = size + alignment > (1u << 20) ? ((size + alignment + 0xffff) & ~0xffffull) : (1u << 20);
    auto* p = static_cast<uint8_t*>(VirtualAlloc(nullptr, size_t(n), MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (!p) return nullptr;
    memory::ensure(uint64_t(p), n);
    chunks_.push_back({p, n, 0});
    auto& c = chunks_.back();
    c.used = size;  // VirtualAlloc memory is 64 KiB aligned
    return c.p;
}

namespace stream {
namespace {

Stream* S(void* list) { return list::stream_of(list); }
using namespace cmd;

void WINAPI draw(void* l, uint32_t c, uint32_t i, uint32_t f, uint32_t fi) {
    if (auto* s = S(l)) s->add(Op::Draw, Draw{c, i, f, fi});
    stats::add(stats::Draws);
}
void WINAPI draw_indexed(void* l, uint32_t c, uint32_t i, uint32_t f, int32_t b, uint32_t fi) {
    if (auto* s = S(l)) s->add(Op::DrawIndexed, DrawIndexed{c, i, f, b, fi});
    stats::add(stats::Draws);
}
void WINAPI dispatch(void* l, uint32_t x, uint32_t y, uint32_t z) {
    if (auto* s = S(l)) s->add(Op::Dispatch, Dispatch{x, y, z, 0});
    stats::add(stats::Dispatches);
}
void WINAPI dispatch_x(void* l, uint32_t x, uint32_t y, uint32_t z, uint32_t f) {
    if (auto* s = S(l)) s->add(Op::Dispatch, Dispatch{x, y, z, f});
    stats::add(stats::Dispatches);
}
void WINAPI dispatch_mesh(void* l, uint32_t x, uint32_t y, uint32_t z) {
    if (auto* s = S(l)) s->add(Op::DispatchMesh, Dispatch{x, y, z, 0});
    stats::add(stats::Draws);
}
void WINAPI dispatch_mesh_x(void* l, uint32_t x, uint64_t flags, uint64_t indirect, uint64_t payload) {
    stats::add(stats::Draws);
    if (indirect) {
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) KL_WARN("gpu", "mesh dispatches with GPU-side group counts are not run yet (flags %llx, payload %llx)", (unsigned long long)flags, (unsigned long long)payload);
        return;
    }
    if (auto* s = S(l)) s->add(Op::DispatchMesh, Dispatch{x, 1, 1, uint32_t(flags)});
}
uint64_t buffer_address(void* res, uint64_t offset) {
    resource::Info r;
    return resource::info(res, r) ? r.address + offset : 0;
}
void WINAPI execute_indirect(void* l, void* sig, uint32_t max, void* args, uint64_t off, void* count, uint64_t coff) {
    if (auto* s = S(l))
        s->add(Op::ExecuteIndirect, Indirect{sig, buffer_address(args, off), off, max, count ? buffer_address(count, coff) : 0, coff});
    stats::add(stats::Draws);
}

void WINAPI copy_buffer_region(void* l, void* dst, uint64_t doff, void* src, uint64_t soff, uint64_t len) {
    if (auto* s = S(l)) s->add(Op::CopyBuffer, CopyBuffer{buffer_address(dst, doff), buffer_address(src, soff), len});
    stats::add(stats::Copies);
}
void WINAPI copy_memory(void* l, uint64_t dst, uint64_t src, uint64_t len, uint32_t) {
    if (auto* s = S(l)) s->add(Op::CopyMem, CopyBuffer{dst, src, len});
    stats::add(stats::Copies);
}
void WINAPI copy_texture_region(void* l, const CopyLocation* dst, uint32_t x, uint32_t y, uint32_t z, const CopyLocation* src,
                                const uint32_t* box) {
    auto* s = S(l);
    if (s && dst && src) {
        CopyTexture c{*dst, *src, x, y, z, box ? 1u : 0u, {}};
        if (box) memcpy(c.box, box, sizeof c.box);
        s->add(Op::CopyTexture, c);
    }
    stats::add(stats::Copies);
}
void WINAPI copy_resource(void* l, void* dst, void* src) {
    if (auto* s = S(l)) s->add(Op::CopyResource, CopyResource{dst, src});
    stats::add(stats::Copies);
}
void WINAPI write32(void* l, uint64_t address, uint32_t value, uint32_t, uint32_t flags) {
    if (auto* s = S(l)) s->add(Op::Write32, Write{address, value, 4, flags});
}
void WINAPI write64(void* l, uint64_t address, uint64_t value, uint32_t, uint32_t flags) {
    if (auto* s = S(l)) s->add(Op::Write64, Write{address, value, 8, flags});
}
void WINAPI wait_value(void* l, uint64_t address, uint32_t cmp, uint32_t ref, uint32_t mask, uint32_t) {
    if (auto* s = S(l)) s->add(Op::WaitValue, WaitValue{address, cmp, ref, 4, mask});
}
void WINAPI wait_value64(void* l, uint64_t address, uint32_t cmp, uint64_t ref, uint32_t) {
    if (auto* s = S(l)) s->add(Op::WaitValue, WaitValue{address, cmp, ref, 8, 0});
}
void WINAPI resolve_queries(void* l, void*, uint32_t, uint32_t, uint32_t count, void* dst, uint64_t off) {
    if (auto* s = S(l)) s->add(Op::ResolveQueries, ResolveQueries{buffer_address(dst, off), count});
}
void WINAPI barrier(void* l, uint32_t n, const void*) {
    if (auto* s = S(l)) s->add(Op::Barrier, n);
    stats::add(stats::Barriers);
}
void WINAPI barrier_groups(void* l, uint32_t n, const void*) {
    if (auto* s = S(l)) s->add(Op::Barrier, n);
    stats::add(stats::Barriers);
}
void* WINAPI allocate_upload(void* l, uint64_t size, uint64_t alignment) {
    auto* s = S(l);
    return s ? s->upload(size, alignment) : nullptr;
}
uint64_t WINAPI driver_hint(void*, uint64_t, uint64_t) { return 0; }

template <Op OP>
void WINAPI state1(void* l, uint64_t a) {
    if (auto* s = S(l)) s->add(OP, a);
    stats::add(stats::OtherCommands);
}
// Root arguments: G = 1 graphics, 0 compute.
template <uint32_t G>
void WINAPI root_signature(void* l, void* r) {
    if (auto* s = S(l)) s->add(Op::RootSignature, Root{G, 0, uint64_t(r)});
}
template <uint32_t G>
void WINAPI root_table(void* l, uint32_t slot, uint64_t handle) {
    if (auto* s = S(l)) s->add(Op::RootTable, Root{G, slot, handle});
}
template <uint32_t G>
void WINAPI root_view(void* l, uint32_t slot, uint64_t address) {
    if (auto* s = S(l)) s->add(Op::RootView, Root{G, slot, address});
}
template <uint32_t G>
void WINAPI root_constant(void* l, uint32_t slot, uint32_t value, uint32_t offset) {
    if (auto* s = S(l)) {
        RootConstants c{G, slot, offset, 1, {value}};
        s->add(Op::RootConstants, &c, uint32_t(offsetof(RootConstants, values) + 4));
    }
}
template <uint32_t G>
void WINAPI root_constants(void* l, uint32_t slot, uint32_t n, const uint32_t* data, uint32_t first) {
    auto* s = S(l);
    if (!s || !data) return;
    RootConstants c{G, slot, first, n > 64 ? 64 : n, {}};
    memcpy(c.values, data, c.count * 4);
    s->add(Op::RootConstants, &c, uint32_t(offsetof(RootConstants, values) + 4 * c.count));
}
void WINAPI index_buffer(void* l, const IndexBuffer* v) {
    if (auto* s = S(l)) s->add(Op::IndexBuffer, v ? *v : IndexBuffer{});
}
void WINAPI vertex_buffers(void* l, uint32_t start, uint32_t n, const VertexView* views) {
    auto* s = S(l);
    if (!s) return;
    VertexBuffers v{start, n > 32 ? 32 : n, {}};
    if (views) memcpy(v.views, views, v.count * sizeof(VertexView));
    s->add(Op::VertexBuffers, &v, uint32_t(offsetof(VertexBuffers, views) + v.count * sizeof(VertexView)));
}
void WINAPI viewports(void* l, uint32_t n, const float* items) {
    auto* s = S(l);
    if (!s || !items) return;
    Viewports v{n > 16 ? 16 : n, 0, {}};
    memcpy(v.v, items, v.count * 24);
    s->add(Op::Viewports, v);
}
void WINAPI scissors(void* l, uint32_t n, const int32_t* items) {
    auto* s = S(l);
    if (!s || !items) return;
    Scissors v{n > 16 ? 16 : n, 0, {}};
    memcpy(v.r, items, v.count * 16);
    s->add(Op::Scissors, v);
}
void WINAPI heaps(void* l, uint32_t n, void* const* items) {
    auto* s = S(l);
    if (!s || !items) return;
    Heaps h{n > 8 ? 8 : n, 0, {}};
    memcpy(h.heaps, items, h.count * 8);
    s->add(Op::Heaps, h);
}
void WINAPI blend_factor(void* l, const float* f) {
    float v[4] = {};
    if (f) memcpy(v, f, sizeof v);
    if (auto* s = S(l)) s->add(Op::BlendFactor, v);
}
void WINAPI clear_rtv(void* l, uint64_t handle, const float* color, uint32_t, const void*) {
    ClearRtv p{handle, {}};
    if (color) memcpy(p.color, color, sizeof p.color);
    if (auto* s = S(l)) s->add(Op::ClearRtv, p);
}
void WINAPI clear_dsv(void* l, uint64_t handle, uint32_t flags, float depth, uint8_t stencil, uint32_t, const void*) {
    if (auto* s = S(l)) s->add(Op::ClearDsv, ClearDsv{handle, flags, depth, stencil});
}
template <Op OP>
void WINAPI clear_uav(void* l, uint64_t gpu, uint64_t cpu, void* res, const uint32_t* values, uint32_t, const void*) {
    ClearUav c{gpu, cpu, uint64_t(res), {}};
    if (values) memcpy(c.values, values, 16);
    if (auto* s = S(l)) s->add(OP, c);
}
void WINAPI targets(void* l, uint32_t n, const uint64_t* rtvs, BOOL single, const uint64_t* dsv) {
    auto* s = S(l);
    if (!s) return;
    Targets t{n > 8 ? 8 : n, uint32_t(single), {}, dsv ? *dsv : 0};
    for (uint32_t i = 0; i < t.count; ++i) t.rtv[i] = rtvs ? (single ? rtvs[0] + uint64_t(i) * 128 : rtvs[i]) : 0;
    s->add(Op::Targets, t);
}
void WINAPI ignored(void*) {}

}  // namespace

void fill_list_methods(void** t, bool newer) {
    auto at = [&](size_t slot, void* fn) { t[newer && slot >= 59 ? slot + 2 : slot] = fn; };
    at(12, (void*)&draw);
    at(13, (void*)&draw_indexed);
    at(14, (void*)&dispatch);
    at(15, (void*)&copy_buffer_region);
    at(16, (void*)&copy_texture_region);
    at(17, (void*)&copy_resource);
    at(19, (void*)&state1<Op::Resolve>);
    at(20, (void*)&state1<Op::Topology>);
    at(21, (void*)&viewports);
    at(22, (void*)&scissors);
    at(23, (void*)&blend_factor);
    at(24, (void*)&state1<Op::StencilRef>);
    at(25, (void*)&state1<Op::SetPipeline>);
    at(26, (void*)&barrier);
    at(28, (void*)&ignored);
    at(29, (void*)&ignored);
    at(30, (void*)&resolve_queries);
    at(32, (void*)&heaps);
    at(33, (void*)&root_signature<0>);
    at(34, (void*)&root_signature<1>);
    at(35, (void*)&root_table<0>);
    at(36, (void*)&root_table<1>);
    at(37, (void*)&root_constant<0>);
    at(38, (void*)&root_constant<1>);
    at(39, (void*)&root_constants<0>);
    at(40, (void*)&root_constants<1>);
    for (size_t s = 41; s <= 46; s += 2) at(s, (void*)&root_view<0>), at(s + 1, (void*)&root_view<1>);
    at(47, (void*)&index_buffer);
    at(48, (void*)&vertex_buffers);
    at(50, (void*)&targets);
    at(51, (void*)&clear_dsv);
    at(52, (void*)&clear_rtv);
    at(53, (void*)&clear_uav<Op::ClearUavUint>);
    at(54, (void*)&clear_uav<Op::ClearUavFloat>);
    at(55, (void*)&state1<Op::Discard>);
    at(59, (void*)&execute_indirect);
    at(60, (void*)&copy_memory);
    at(68, (void*)&ignored);
    at(69, (void*)&write32);
    at(70, (void*)&write64);
    at(71, (void*)&wait_value);
    at(72, (void*)&wait_value64);
    at(78, (void*)&dispatch_x);
    at(83, (void*)&copy_texture_region);
    at(84, (void*)&copy_resource);
    at(98, (void*)&dispatch_mesh_x);
    at(99, (void*)&driver_hint);
    at(127, (void*)&dispatch_mesh);
    if (newer) {
        t[142] = (void*)&ignored;
        t[146] = (void*)&barrier_groups;
        t[148] = (void*)&allocate_upload;
    }
}

void run_cpu_side(const Stream& s) {
    using namespace cmd;
    s.each([](Op op, const uint8_t* p, uint32_t) {
        switch (op) {
            case Op::Write32:
            case Op::Write64: {
                Write w;
                memcpy(&w, p, sizeof w);
                if (!w.address) break;
                uint64_t v = w.value;
                if (op == Op::Write64 && w.flags == 1) {
                    static auto start = std::chrono::steady_clock::now();
                    v = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count() / 10);
                } else if (op == Op::Write64 && w.flags == 2) {
                    LARGE_INTEGER q;
                    QueryPerformanceCounter(&q);
                    v = uint64_t(q.QuadPart);
                }
                if (w.bytes == 4) *reinterpret_cast<volatile uint32_t*>(w.address) = uint32_t(v);
                else *reinterpret_cast<volatile uint64_t*>(w.address) = v;
                break;
            }
            case Op::CopyBuffer:
            case Op::CopyMem: {
                CopyBuffer c;
                memcpy(&c, p, sizeof c);
                if (c.dst && c.src && c.len) memmove(reinterpret_cast<void*>(c.dst), reinterpret_cast<void*>(c.src), size_t(c.len));
                break;
            }
            case Op::ResolveQueries: {
                ResolveQueries r;
                memcpy(&r, p, sizeof r);
                if (r.dst) memset(reinterpret_cast<void*>(r.dst), 0, size_t(r.count) * 8);
                break;
            }
            default: break;
        }
    });
}

}  // namespace stream
}  // namespace kl::gpu
