// keylime GPU: list command recorders. Slot map and argument shapes after
// VIctory (list.rs); recording + CPU-side execution are keylime's own.
#include "gpu/stream.h"

#include <windows.h>

#include <chrono>
#include <cstring>

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
    auto block = std::make_unique<uint8_t[]>(size_t(size + alignment));
    auto p = reinterpret_cast<uintptr_t>(block.get());
    p = (p + alignment - 1) & ~(alignment - 1);
    uploads_.push_back(std::move(block));
    return reinterpret_cast<void*>(p);
}

namespace stream {
namespace {

Stream* S(void* list) { return list::stream_of(list); }

struct DrawArgs { uint32_t count, instances, first, first_instance; };
struct DrawIndexedArgs { uint32_t count, instances, first; int32_t base; uint32_t first_instance; };
struct DispatchArgs { uint32_t x, y, z, flags; };
struct CopyBufferArgs { uint64_t dst, src, len; };
struct WriteArgs { uint64_t address, value; uint32_t bytes, flags; };
struct ResolveQueriesArgs { uint64_t dst; uint32_t count; };
struct U32x4 { uint32_t a, b, c, d; };
struct U64x4 { uint64_t a, b, c, d; };

void WINAPI draw(void* l, uint32_t c, uint32_t i, uint32_t f, uint32_t fi) {
    if (auto* s = S(l)) s->add(Op::Draw, DrawArgs{c, i, f, fi});
    stats::add(stats::Draws);
}
void WINAPI draw_indexed(void* l, uint32_t c, uint32_t i, uint32_t f, int32_t b, uint32_t fi) {
    if (auto* s = S(l)) s->add(Op::DrawIndexed, DrawIndexedArgs{c, i, f, b, fi});
    stats::add(stats::Draws);
}
void WINAPI dispatch(void* l, uint32_t x, uint32_t y, uint32_t z) {
    if (auto* s = S(l)) s->add(Op::Dispatch, DispatchArgs{x, y, z, 0});
    stats::add(stats::Dispatches);
}
void WINAPI dispatch_x(void* l, uint32_t x, uint32_t y, uint32_t z, uint32_t f) {
    if (auto* s = S(l)) s->add(Op::Dispatch, DispatchArgs{x, y, z, f});
    stats::add(stats::Dispatches);
}
void WINAPI dispatch_mesh(void* l, uint32_t x, uint32_t y, uint32_t z) {
    if (auto* s = S(l)) s->add(Op::DispatchMesh, DispatchArgs{x, y, z, 0});
    stats::add(stats::Draws);
}
void WINAPI dispatch_mesh_x(void* l, uint32_t x, uint32_t y, uint32_t z, uint32_t f) {
    if (auto* s = S(l)) s->add(Op::DispatchMesh, DispatchArgs{x, y, z, f});
    stats::add(stats::Draws);
}
void WINAPI execute_indirect(void* l, void* sig, uint32_t max, void* args, uint64_t off, void*, uint64_t) {
    if (auto* s = S(l)) s->add(Op::ExecuteIndirect, U64x4{uint64_t(sig), uint64_t(args), off, max});
    stats::add(stats::Draws);
}

uint64_t buffer_address(void* res, uint64_t offset) {
    resource::Info r;
    return resource::info(res, r) ? r.address + offset : 0;
}
void WINAPI copy_buffer_region(void* l, void* dst, uint64_t doff, void* src, uint64_t soff, uint64_t len) {
    if (auto* s = S(l)) s->add(Op::CopyBuffer, CopyBufferArgs{buffer_address(dst, doff), buffer_address(src, soff), len});
    stats::add(stats::Copies);
}
void WINAPI copy_memory(void* l, uint64_t dst, uint64_t src, uint64_t len, uint32_t) {
    if (auto* s = S(l)) s->add(Op::CopyMem, CopyBufferArgs{dst, src, len});
    stats::add(stats::Copies);
}
void WINAPI copy_texture_region(void* l, const void* dst, uint32_t x, uint32_t y, uint32_t z, const void* src, const void*) {
    if (auto* s = S(l)) s->add(Op::CopyTexture, U64x4{uint64_t(dst), uint64_t(src), uint64_t(x) | uint64_t(y) << 32, z});
    stats::add(stats::Copies);
}
void WINAPI copy_resource(void* l, void* dst, void* src) {
    if (auto* s = S(l)) s->add(Op::CopyResource, U64x4{uint64_t(dst), uint64_t(src), 0, 0});
    stats::add(stats::Copies);
}
void WINAPI write32(void* l, uint64_t address, uint32_t value, uint32_t, uint32_t flags) {
    if (auto* s = S(l)) s->add(Op::Write32, WriteArgs{address, value, 4, flags});
}
void WINAPI write64(void* l, uint64_t address, uint64_t value, uint32_t, uint32_t flags) {
    if (auto* s = S(l)) s->add(Op::Write64, WriteArgs{address, value, 8, flags});
}
void WINAPI wait_value(void* l, uint64_t address, uint32_t cmp, uint64_t ref) {
    if (auto* s = S(l)) s->add(Op::WaitValue, U64x4{address, cmp, ref, 0});
}
void WINAPI resolve_queries(void* l, void*, uint32_t, uint32_t, uint32_t count, void* dst, uint64_t off) {
    if (auto* s = S(l)) s->add(Op::ResolveQueries, ResolveQueriesArgs{buffer_address(dst, off), count});
}
void WINAPI barrier(void* l, uint32_t n, const void*) {
    if (auto* s = S(l)) s->add(Op::Barrier, U32x4{n, 0, 0, 0});
    stats::add(stats::Barriers);
}
void WINAPI barrier_groups(void* l, uint32_t n, const void*) {
    if (auto* s = S(l)) s->add(Op::Barrier, U32x4{n, 1, 0, 0});
    stats::add(stats::Barriers);
}
void* WINAPI allocate_upload(void* l, uint64_t size, uint64_t alignment) {
    auto* s = S(l);
    return s ? s->upload(size, alignment) : nullptr;
}
uint64_t WINAPI driver_hint(void*, uint64_t, uint64_t) { return 0; }

template <Op OP>
void WINAPI state1(void* l, uint64_t a) {
    if (auto* s = S(l)) s->add(OP, U64x4{a, 0, 0, 0});
    stats::add(stats::OtherCommands);
}
template <Op OP>
void WINAPI state3(void* l, uint64_t a, uint64_t b, uint64_t c) {
    if (auto* s = S(l)) s->add(OP, U64x4{a, b, c, 0});
    stats::add(stats::OtherCommands);
}
template <Op OP, size_t ELEM>
void WINAPI array_state(void* l, uint32_t n, const void* items) {
    auto* s = S(l);
    if (s && items && n <= 64) {
        static thread_local uint8_t buf[8 + 64 * 48];
        memcpy(buf, &n, 4);
        memcpy(buf + 8, items, n * ELEM);
        s->add(OP, buf, uint32_t(8 + n * ELEM));
    }
    stats::add(stats::OtherCommands);
}
void WINAPI blend_factor(void* l, const float* f) {
    float v[4] = {};
    if (f) memcpy(v, f, sizeof v);
    if (auto* s = S(l)) s->add(Op::BlendFactor, v);
}
void WINAPI set_constants(void* l, uint32_t param, uint32_t n, const void* data, uint32_t first) {
    auto* s = S(l);
    if (s && data && n <= 64) {
        uint32_t buf[3 + 64] = {param, n, first};
        memcpy(buf + 3, data, n * 4);
        s->add(Op::RootConstants, buf, (3 + n) * 4);
    }
}
struct ClearRtvArgs { uint64_t h; float c[4]; };
void WINAPI clear_rtv(void* l, uint64_t handle, const float* color, uint32_t, const void*) {
    ClearRtvArgs p{handle, {}};
    if (color) memcpy(p.c, color, sizeof p.c);
    if (auto* s = S(l)) s->add(Op::ClearRtv, p);
}
struct ClearDsvArgs { uint64_t h; uint32_t f; float d; uint32_t s; };
void WINAPI clear_dsv(void* l, uint64_t handle, uint32_t flags, float depth, uint8_t stencil, uint32_t, const void*) {
    if (auto* s = S(l)) s->add(Op::ClearDsv, ClearDsvArgs{handle, flags, depth, stencil});
}
void WINAPI targets(void* l, uint32_t n, const uint64_t* rtvs, BOOL single, const uint64_t* dsv) {
    auto* s = S(l);
    if (!s) return;
    uint64_t buf[11] = {n, uint64_t(single)};
    for (uint32_t i = 0; i < n && i < 8; ++i) buf[2 + i] = rtvs ? (single ? rtvs[0] + uint64_t(i) * 128 : rtvs[i]) : 0;
    buf[10] = dsv ? *dsv : 0;
    s->add(Op::Targets, buf, sizeof buf);
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
    at(21, (void*)&array_state<Op::Viewports, 24>);
    at(22, (void*)&array_state<Op::Scissors, 16>);
    at(23, (void*)&blend_factor);
    at(24, (void*)&state1<Op::StencilRef>);
    at(25, (void*)&state1<Op::SetPipeline>);
    at(26, (void*)&barrier);
    at(28, (void*)&ignored);
    at(29, (void*)&ignored);
    at(30, (void*)&resolve_queries);
    at(32, (void*)&array_state<Op::Heaps, 8>);
    at(33, (void*)&state1<Op::RootSignature>);
    at(34, (void*)&state1<Op::RootSignature>);
    at(35, (void*)&state3<Op::RootTable>);
    at(36, (void*)&state3<Op::RootTable>);
    at(37, (void*)&state3<Op::RootConstant>);
    at(38, (void*)&state3<Op::RootConstant>);
    at(39, (void*)&set_constants);
    at(40, (void*)&set_constants);
    for (size_t s = 41; s <= 46; ++s) at(s, (void*)&state3<Op::RootView>);
    at(47, (void*)&state1<Op::IndexBuffer>);
    at(48, (void*)&state3<Op::VertexBuffers>);
    at(50, (void*)&targets);
    at(51, (void*)&clear_dsv);
    at(52, (void*)&clear_rtv);
    at(53, (void*)&state3<Op::ClearUavUint>);
    at(54, (void*)&state3<Op::ClearUavFloat>);
    at(55, (void*)&state1<Op::Discard>);
    at(59, (void*)&execute_indirect);
    at(60, (void*)&copy_memory);
    at(68, (void*)&ignored);
    at(69, (void*)&write32);
    at(70, (void*)&write64);
    at(71, (void*)&wait_value);
    at(72, (void*)&wait_value);
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
    s.each([](Op op, const uint8_t* p, uint32_t) {
        switch (op) {
            case Op::Write32:
            case Op::Write64: {
                WriteArgs w;
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
                CopyBufferArgs c;
                memcpy(&c, p, sizeof c);
                if (c.dst && c.src && c.len) memmove(reinterpret_cast<void*>(c.dst), reinterpret_cast<void*>(c.src), size_t(c.len));
                break;
            }
            case Op::ResolveQueries: {
                ResolveQueriesArgs r;
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
