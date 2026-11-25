// keylime shaders: the console's rectangle lists (topology 17, hardware
// DI_PT_RECTLIST). Each rectangle comes as three vertices v0, v1, v2 with
// v0 the corner both edges share; the hardware makes the fourth as
// v1 + v2 - v0, for the position and every attribute. On the host this is a
// geometry shader taking each triangle and emitting a 4-vertex strip.
#include "shader/rectlist.h"

#include <array>

#include "shader/spirv.h"

namespace kl::shader {
using sp::Id;

// quad: quad lists (topology 19) arrive as 4-vertex primitives (lines with adjacency) and leave as v0 v1 v3 v2.
std::vector<uint32_t> rect_geometry(const std::vector<uint8_t>& locations, bool quad) {
    const uint32_t N = quad ? 4 : 3;
    sp::Builder b;
    b.capability(spv::CapabilityShader);
    b.capability(spv::CapabilityGeometry);
    b.capability(spv::CapabilityPhysicalStorageBufferAddresses);
    Id t_void = b.t_void(), t_f32 = b.t_f32(), t_v4 = b.t_vec(t_f32, 4), t_u32 = b.t_u32();
    Id t_fn = b.t_fn(t_void, {});
    // gl_in[3].gl_Position and gl_Position out, as blocks.
    Id in_block = b.t_struct({t_v4});
    b.decorate(in_block, spv::DecorationBlock);
    b.member_decorate(in_block, 0, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInPosition)});
    Id in_arr = b.t_array(in_block, N);
    Id in_pos = b.variable(b.t_ptr(spv::StorageClassInput, in_arr), spv::StorageClassInput);
    Id out_block = b.t_struct({t_v4});
    b.decorate(out_block, spv::DecorationBlock);
    b.member_decorate(out_block, 0, spv::DecorationBuiltIn, {uint32_t(spv::BuiltInPosition)});
    Id out_pos = b.variable(b.t_ptr(spv::StorageClassOutput, out_block), spv::StorageClassOutput);
    std::vector<Id> interface = {in_pos, out_pos};
    Id p_in_v4 = b.t_ptr(spv::StorageClassInput, t_v4), p_out_v4 = b.t_ptr(spv::StorageClassOutput, t_v4);
    Id arr_v4 = b.t_array(t_v4, N);
    std::vector<std::pair<Id, Id>> vars;  // (in array, out)
    for (uint8_t loc : locations) {
        Id i = b.variable(b.t_ptr(spv::StorageClassInput, arr_v4), spv::StorageClassInput);
        Id o = b.variable(p_out_v4, spv::StorageClassOutput);
        b.decorate(i, spv::DecorationLocation, {loc});
        b.decorate(o, spv::DecorationLocation, {loc});
        interface.push_back(i), interface.push_back(o);
        vars.push_back({i, o});
    }
    Id fn = b.id();
    b.begin_function(fn, t_void, t_fn);
    b.label();
    auto at = [&](Id arr, uint32_t k, bool block) {
        if (block) return b.op(spv::OpAccessChain, p_in_v4, {arr, b.c_u32(k), b.c_u32(0)});
        return b.op(spv::OpAccessChain, p_in_v4, {arr, b.c_u32(k)});
    };
    auto load = [&](Id ptr) { return b.op(spv::OpLoad, t_v4, {ptr}); };
    Id pos[4], val[16][4];
    for (uint32_t k = 0; k < N; ++k) pos[k] = load(at(in_pos, k, true));
    for (size_t v = 0; v < vars.size() && v < 16; ++v)
        for (uint32_t k = 0; k < N; ++k) val[v][k] = load(at(vars[v].first, k, false));
    Id pos_out = b.op(spv::OpAccessChain, p_out_v4, {out_pos, b.c_u32(0)});
    auto emit = [&](Id p, const std::vector<Id>& vs) {
        b.op_void(spv::OpStore, {pos_out, p});
        for (size_t v = 0; v < vs.size(); ++v) b.op_void(spv::OpStore, {vars[v].second, vs[v]});
        b.op_void(spv::OpEmitVertex, {});
    };
    auto fourth = [&](Id a0, Id a1, Id a2) { return b.op(spv::OpFSub, t_v4, {b.op(spv::OpFAdd, t_v4, {a1, a2}), a0}); };
    size_t n = vars.size() < 16 ? vars.size() : 16;
    if (quad) {
        for (uint32_t k : {0u, 1u, 3u, 2u}) {
            std::vector<Id> vs;
            for (size_t v = 0; v < n; ++v) vs.push_back(val[v][k]);
            emit(pos[k], vs);
        }
    } else {
        // The right-angle corner is the vertex opposite the longest edge (titles use either v0 or v1).
        Id t_bool = b.t_bool();
        auto comp = [&](Id v, uint32_t i) { return b.op(spv::OpCompositeExtract, t_f32, {v, i}); };
        auto ndc = [&](Id p, uint32_t i) { return b.op(spv::OpFDiv, t_f32, {comp(p, i), comp(p, 3)}); };
        auto len2 = [&](Id p, Id q) {
            Id dx = b.op(spv::OpFSub, t_f32, {ndc(p, 0), ndc(q, 0)}), dy = b.op(spv::OpFSub, t_f32, {ndc(p, 1), ndc(q, 1)});
            return b.op(spv::OpFAdd, t_f32, {b.op(spv::OpFMul, t_f32, {dx, dx}), b.op(spv::OpFMul, t_f32, {dy, dy})});
        };
        Id l01 = len2(pos[0], pos[1]), l12 = len2(pos[1], pos[2]), l20 = len2(pos[2], pos[0]);
        auto ge = [&](Id x, Id y) { return b.op(spv::OpFOrdGreaterThanEqual, t_bool, {x, y}); };
        auto band = [&](Id x, Id y) { return b.op(spv::OpLogicalAnd, t_bool, {x, y}); };
        auto bnot = [&](Id x) { return b.op(spv::OpLogicalNot, t_bool, {x}); };
        Id c1 = band(ge(l20, l01), ge(l20, l12));
        Id c2 = band(bnot(c1), ge(l01, l12));
        Id c0 = band(bnot(c1), bnot(c2));
        auto w = [&](Id c) { return b.op(spv::OpSelect, t_f32, {c, b.c_f32(1.0f), b.c_f32(0.0f)}); };
        Id W[3] = {w(c0), w(c1), w(c2)};
        // Roles in winding order: a (before the corner), c (the corner), b (after it).
        Id wc[3] = {W[0], W[1], W[2]}, wa[3] = {W[1], W[2], W[0]}, wb[3] = {W[2], W[0], W[1]};
        auto pick = [&](const Id* v, const Id* wt) {
            Id r = b.op(spv::OpVectorTimesScalar, t_v4, {v[0], wt[0]});
            r = b.op(spv::OpFAdd, t_v4, {r, b.op(spv::OpVectorTimesScalar, t_v4, {v[1], wt[1]})});
            return b.op(spv::OpFAdd, t_v4, {r, b.op(spv::OpVectorTimesScalar, t_v4, {v[2], wt[2]})});
        };
        auto roles = [&](const Id* v, Id out[4]) {
            Id A = pick(v, wa), C = pick(v, wc), B = pick(v, wb);
            out[0] = A, out[1] = C, out[2] = b.op(spv::OpFSub, t_v4, {b.op(spv::OpFAdd, t_v4, {A, B}), C}), out[3] = B;
        };
        Id P[4];
        roles(pos, P);
        std::vector<std::array<Id, 4>> V(n);
        for (size_t v = 0; v < n; ++v) {
            Id r[4];
            roles(val[v], r);
            V[v] = {r[0], r[1], r[2], r[3]};
        }
        // Strip a, c, d, b: triangles (a, c, d) and (d, c, b), both in the original winding.
        for (uint32_t k = 0; k < 4; ++k) {
            std::vector<Id> vs;
            for (size_t v = 0; v < n; ++v) vs.push_back(V[v][k]);
            emit(P[k], vs);
        }
    }
    b.op_void(spv::OpEndPrimitive, {});
    b.op_void(spv::OpReturn, {});
    b.end_function();
    b.entry(spv::ExecutionModelGeometry, fn, "main", interface);
    b.mode(fn, quad ? spv::ExecutionModeInputLinesAdjacency : spv::ExecutionModeTriangles);
    b.mode(fn, spv::ExecutionModeOutputTriangleStrip);
    b.mode(fn, spv::ExecutionModeOutputVertices, {4});
    b.mode(fn, spv::ExecutionModeInvocations, {1});
    (void)t_u32;
    return b.finish();
}

}  // namespace kl::shader
